from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HueFitNetV2(nn.Module):
    def __init__(
        self,
        visual_dim: int = 768,
        color_dim: int = 13,
        num_cats: int = 160,
        num_sem_cats: int = 16,
        d_model: int = 160,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
        visual_dropout: float = 0.35,
        visual_noise_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.color_dim = color_dim
        self.num_cats = num_cats
        self.num_sem_cats = num_sem_cats
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.visual_noise_std = visual_noise_std

        self.visual_bottleneck = nn.Sequential(
            nn.Linear(visual_dim, 128),
            nn.LayerNorm(128),
            nn.Dropout(visual_dropout),
            nn.GELU(),
        )
        self.color_proj = nn.Sequential(
            nn.Linear(color_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.cat_emb = nn.Embedding(num_cats, 24, padding_idx=0)
        self.sem_emb = nn.Embedding(num_sem_cats, 16, padding_idx=0)
        self.meta_proj = nn.Sequential(
            nn.Linear(2, 16),
            nn.LayerNorm(16),
            nn.GELU(),
        )

        self.token_fuse = nn.Sequential(
            nn.Linear(248, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.post_norm = nn.LayerNorm(d_model)

        self.summary_proj = nn.Sequential(
            nn.Linear(20, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.outfit_head = nn.Sequential(
            nn.Linear(d_model * 3 + 48, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        self.item_head = nn.Sequential(
            nn.Linear(d_model * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _compute_set_summary(
        visual_emb: torch.Tensor,
        color_feats: torch.Tensor,
        formality: torch.Tensor,
        pad_mask: torch.Tensor,
        has_visual: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = color_feats.shape
        valid = (~pad_mask).float()
        valid_count = valid.sum(dim=1).clamp(min=1.0)

        oklab = color_feats[:, :, 0:3]
        L_val = oklab[:, :, 0]
        C_val = color_feats[:, :, 3]

        pair_valid = valid.unsqueeze(2) * valid.unsqueeze(1)
        eye = torch.eye(L, device=color_feats.device, dtype=torch.float32).unsqueeze(0)
        pair_mask = pair_valid * (1.0 - eye)
        pair_count = pair_mask.sum(dim=(1, 2)).clamp(min=1.0)

        diff_ok = oklab.unsqueeze(2) - oklab.unsqueeze(1)
        de = torch.sqrt((diff_ok * diff_ok).sum(dim=-1) + 1e-8)
        dL = torch.abs(diff_ok[:, :, :, 0])
        dab = torch.sqrt(diff_ok[:, :, :, 1] ** 2 + diff_ok[:, :, :, 2] ** 2 + 1e-8)

        de_mean = (de * pair_mask).sum(dim=(1, 2)) / pair_count
        de_var = (((de - de_mean.view(B, 1, 1)) ** 2) * pair_mask).sum(
            dim=(1, 2)
        ) / pair_count
        de_std = torch.sqrt(de_var + 1e-8)
        has_pairs = pair_mask.sum(dim=(1, 2)) > 0
        de_min = torch.where(
            has_pairs,
            de.masked_fill(pair_mask < 0.5, 1e4).amin(dim=(1, 2)),
            torch.zeros_like(de_mean),
        )
        de_max = torch.where(
            has_pairs,
            de.masked_fill(pair_mask < 0.5, -1e4).amax(dim=(1, 2)),
            torch.zeros_like(de_mean),
        )

        dL_mean = (dL * pair_mask).sum(dim=(1, 2)) / pair_count
        dL_max = torch.where(
            has_pairs,
            dL.masked_fill(pair_mask < 0.5, -1e4).amax(dim=(1, 2)),
            torch.zeros_like(dL_mean),
        )

        dab_mean = (dab * pair_mask).sum(dim=(1, 2)) / pair_count
        dab_max = torch.where(
            has_pairs,
            dab.masked_fill(pair_mask < 0.5, -1e4).amax(dim=(1, 2)),
            torch.zeros_like(dab_mean),
        )

        L_mean = (L_val * valid).sum(dim=1) / valid_count
        L_std = torch.sqrt(
            (((L_val - L_mean.unsqueeze(1)) ** 2) * valid).sum(dim=1) / valid_count
            + 1e-8
        )
        C_mean = (C_val * valid).sum(dim=1) / valid_count
        neutral_ratio = ((C_val < 0.045).float() * valid).sum(dim=1) / valid_count

        hv = has_visual.float() * valid
        vis_pair_mask = (hv.unsqueeze(2) * hv.unsqueeze(1)) * (1.0 - eye)
        vis_pair_cnt = vis_pair_mask.sum(dim=(1, 2))
        vis_pair_safe = vis_pair_cnt.clamp(min=1.0)
        has_vis_pairs = vis_pair_cnt > 0

        v_norm = F.normalize(visual_emb, dim=-1) * hv.unsqueeze(-1)
        cos_mat = torch.bmm(v_norm, v_norm.transpose(1, 2))
        cos_mean = (cos_mat * vis_pair_mask).sum(dim=(1, 2)) / vis_pair_safe
        cos_var = (((cos_mat - cos_mean.view(B, 1, 1)) ** 2) * vis_pair_mask).sum(
            dim=(1, 2)
        ) / vis_pair_safe
        cos_std = torch.where(
            has_vis_pairs, torch.sqrt(cos_var + 1e-8), torch.zeros_like(cos_mean)
        )
        cos_min = torch.where(
            has_vis_pairs,
            cos_mat.masked_fill(vis_pair_mask < 0.5, 1e4).amin(dim=(1, 2)),
            torch.zeros_like(cos_mean),
        )
        cos_max = torch.where(
            has_vis_pairs,
            cos_mat.masked_fill(vis_pair_mask < 0.5, -1e4).amax(dim=(1, 2)),
            torch.zeros_like(cos_mean),
        )
        cos_mean = torch.where(has_vis_pairs, cos_mean, torch.zeros_like(cos_mean))
        vis_ratio = hv.sum(dim=1) / valid_count

        f_mean = (formality * valid).sum(dim=1) / valid_count
        f_std = torch.sqrt(
            (((formality - f_mean.unsqueeze(1)) ** 2) * valid).sum(dim=1) / valid_count
            + 1e-8
        )
        size_norm = valid_count / 12.0

        return torch.stack(
            [
                de_mean,
                de_min,
                de_max,
                de_std,
                dL_mean,
                dL_max,
                dab_mean,
                dab_max,
                L_mean,
                L_std,
                C_mean,
                neutral_ratio,
                cos_mean,
                cos_min,
                cos_max,
                cos_std,
                vis_ratio,
                f_mean,
                f_std,
                size_norm,
            ],
            dim=-1,
        )

    def forward(
        self,
        visual_emb: torch.Tensor,
        color_feats: torch.Tensor,
        cat_ids: torch.Tensor,
        sem_cats: torch.Tensor,
        formality: torch.Tensor,
        pad_mask: torch.Tensor,
        has_visual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = color_feats.shape
        valid = (~pad_mask).float()

        hv_f = has_visual.float() * valid
        vis_in = visual_emb * hv_f.unsqueeze(-1)
        if self.training and self.visual_noise_std > 0.0:
            noise = torch.randn_like(vis_in) * self.visual_noise_std
            vis_in = (vis_in + noise) * hv_f.unsqueeze(-1)

        v_h = self.visual_bottleneck(vis_in) * hv_f.unsqueeze(-1)
        c_h = self.color_proj(color_feats)
        k_h = self.cat_emb(cat_ids)
        s_h = self.sem_emb(sem_cats)
        m_h = self.meta_proj(torch.stack([formality, hv_f], dim=-1))

        tokens = self.token_fuse(torch.cat([v_h, c_h, k_h, s_h, m_h], dim=-1))
        tokens = tokens * valid.unsqueeze(-1)

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=pad_mask.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        enc_out = self.post_norm(self.encoder(seq, src_key_padding_mask=full_mask))
        cls_out = enc_out[:, 0, :]
        item_out = enc_out[:, 1:, :]

        valid_u = valid.unsqueeze(-1)
        valid_cnt = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = (item_out * valid_u).sum(dim=1) / valid_cnt
        max_pool = item_out.masked_fill(pad_mask.unsqueeze(-1), -1e4).amax(dim=1)

        summary_raw = self._compute_set_summary(
            vis_in, color_feats, formality, pad_mask, has_visual
        )
        summary_h = self.summary_proj(summary_raw)

        outfit_feat = torch.cat([cls_out, mean_pool, max_pool, summary_h], dim=-1)
        outfit_logit = self.outfit_head(outfit_feat).squeeze(-1)

        cls_exp = cls_out.unsqueeze(1).expand(-1, L, -1)
        mean_exp = mean_pool.unsqueeze(1).expand(-1, L, -1)
        item_feat = torch.cat(
            [item_out, cls_exp, torch.abs(item_out - mean_exp)], dim=-1
        )
        item_logits = self.item_head(item_feat).squeeze(-1)

        return outfit_logit, item_logits

    def get_config(self) -> Dict[str, object]:
        return {
            "visual_dim": self.visual_dim,
            "color_dim": self.color_dim,
            "num_cats": self.num_cats,
            "num_sem_cats": self.num_sem_cats,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "visual_noise_std": self.visual_noise_std,
        }
