#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch
import torch.nn as nn

from .vicon_utils import build_alternating_block_lowtri_mask, depatchify, patchify


class Vicon(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        patch_resolution,
        patch_num_in,
        patch_num_out,
        ex_num,
        short_num_min,
        dim_channel,
        dim_token,
    ):
        super().__init__()

        self.patch_resolution = patch_resolution
        self.patch_num_in = patch_num_in
        self.patch_num_out = patch_num_out
        self.ex_num = ex_num + 1  # 1 for qn
        self.short_num_min = short_num_min
        self.dim_channel = dim_channel
        self.dim_token = dim_token

        self.pre_proj = nn.Linear(
            in_features=self.dim_channel * self.patch_resolution**2,
            out_features=self.dim_token,
        )
        self.post_proj = nn.Linear(
            in_features=self.dim_token,
            out_features=self.dim_channel * self.patch_resolution**2,
        )

        self.patch_pos_encoding = nn.Parameter(torch.randn(self.patch_num_in * self.patch_num_in, self.dim_token))
        self.func_pos_encoding = nn.Parameter(torch.randn(self.ex_num * 2, self.dim_token))

        self.transformer = transformer

        mask = (
            1
            - build_alternating_block_lowtri_mask(
                self.ex_num, self.patch_num_in * self.patch_num_in, self.patch_num_out * self.patch_num_out
            )
        ).bool()
        self.register_buffer("mask", mask)

    def forward(self, f, g):
        p = self.patch_num_in
        d = self.dim_token

        # Prepare the pairs (f, g)
        x = torch.cat((f[:, :, None, :, :], g[:, :, None, :, :]), dim=2)  # (bs, pairs, 2, c, h, w)
        bs, pairs, _, c, h, w = x.shape

        feature = x.view(-1, *x.shape[-3:])  # (bs * pairs * 2, c, h, w)
        c, ph, pw = feature.shape[-3:]
        h = ph // p
        w = pw // p
        feature = patchify(feature, patch_num=p)  # (bs * pairs * 2, p * p, c * h * w)

        feature = self.pre_proj(feature)  # (bs * pairs * 2, p * p, d_model)

        feature = feature + self.patch_pos_encoding  # (bs * pairs * 2, p * p, d_model)
        feature = feature.view(bs, -1, p * p, d)  # (bs, pairs * 2, p * p, d_model)

        func_pos_encoding = self.func_pos_encoding.view(1, -1, 1, d)  # (1, cfg["ex_num"] * 2, 1, d_model)
        func_pos_encoding = func_pos_encoding[:, : pairs * 2, :, :]  # (1, pairs * 2, 1, d_model)
        feature = feature + func_pos_encoding  # (bs, pairs * 2, p * p, d_model)
        feature = feature.view(bs, -1, d)  # (bs, pairs * 2 * p * p, d_model)

        mask = self.mask[: pairs * 2 * p * p, : pairs * 2 * p * p]  # (pairs * 2 * p * p, pairs * 2 * p * p)
        feature = self.transformer(feature, mask=mask)  # (bs, pairs * 2 * p * p, d_model)
        feature = feature.view(bs, pairs, 2, p * p, d)  # (bs, pairs, 2, p * p, d_model)
        feature = feature[:, :, 0, :, :]  # (bs, pairs, p * p, d_model) the predicted g

        feature = self.post_proj(feature)  # (bs, pairs, p * p, c * h * w)

        feature = feature.view(bs * pairs, *feature.shape[-2:])  # (bs * pairs, p * p, c * h * w)
        feature = depatchify(feature, patch_num=p, c=c, h=h, w=w)  # (bs * pairs, c, ph, pw)
        feature = feature.view(bs, pairs, *feature.shape[-3:])  # (bs, pairs, c, ph, pw)

        ex_pred = feature[:, :-1, :, :, :]  # (bs, ex_num, c, h, w)
        qn_pred = feature[:, -1:, :, :, :]  # (bs, 1, c, h, w)
        return {"ex_pred": ex_pred, "qn_pred": qn_pred}
