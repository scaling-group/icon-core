#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import einops
import torch.nn as nn
from optree import PyTree

from src.datasets import pytree_utils as pu
from src.models.icon import icon_utils as mu


class ICON(nn.Module):
    """
    Model from:
    - Fine-tune language models as multi-modal differential equation solvers.
    - PDE generalization of in-context operator networks: A study on 1D scalar nonlinear conservation laws.
    """

    def __init__(
        self,
        pre_projection: nn.Module,
        function_pe: nn.Module | None,
        transformer: nn.Module,
        post_projection: nn.Module,
        shot_num_min: int,
        data_mask: bool,
    ):
        super().__init__()
        self.shot_num_min = shot_num_min
        self.data_mask = data_mask
        self.pre_projection = pre_projection
        self.function_pe = function_pe
        self.transformer = transformer
        self.post_projection = post_projection
        self.basic_mask = {}
        self.index_pos = {}
        self.out_mask = {}

    def _get_matrices(self, data: PyTree, mode: str, shot_num_min: int):
        """
        build masks for the model
        """
        data_shape = pu.get_shape(data, exclude_batch=True)
        key = (pu.to_hashable_pytree(data_shape), mode, shot_num_min)
        if key not in self.basic_mask:
            basic_mask, index_pos, out_mask = mu.build_matrices(data_shape, mode=mode, shot_num_min=shot_num_min)
            device = next(self.parameters()).device
            self.basic_mask[key] = basic_mask.to(device)
            self.index_pos[key] = index_pos.to(device)
            self.out_mask[key] = out_mask.to(device)
        return self.basic_mask[key], self.index_pos[key], self.out_mask[key]

    def _basic_forward(self, data, mode, index_pos, basic_mask, shot_num_min, need_weights=False):
        """
        basic forward of the model, with flexibility, no post-processing
        """
        demo_num = data["demo_cond_k"].shape[1]
        cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = mu.build_bool_sequence(
            demo_num, mode=mode, shot_num_min=shot_num_min
        )

        sequence = mu.build_data_sequence(
            data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list
        )  # [batchsize, total_len, dim]
        sequence = self.pre_projection(sequence)  # [batchsize, total_len, model_dim]

        if self.function_pe is not None:
            sequence = sequence + self.function_pe(index_pos)  # [batchsize, total_len, model_dim]
        else:
            pass  # no pe, see https://arxiv.org/pdf/2203.16634

        if self.data_mask:
            # data_mask is used when some data points are masked out.
            # It is not tested in this code, i.e., only data_mask = None is tested
            # So cannot guarantee the correctness
            data_mask = mu.build_data_mask(
                data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list
            )  # [batchsize, total_len]
            data_mask = ~data_mask  # -> zero for "attention", one for "no attention"
        else:
            data_mask = None  # all data is used

        # careful: here basic_mask use zero for "no attention" and one for "attention"
        # but in transformer API, mask should be one for "no attention" and zero for "attention"
        # so we need to invert the mask here
        if need_weights:
            sequence, weights = self.transformer(
                sequence, mask=~basic_mask, src_key_padding_mask=data_mask, need_weights=True
            )
            sequence = self.post_projection(sequence)  # [batchsize, total_len, out_dim]
            return sequence, weights
        else:
            sequence = self.transformer(
                sequence, mask=~basic_mask, src_key_padding_mask=data_mask
            )  # [batchsize, total_len, model_dim]
            sequence = self.post_projection(sequence)  # [batchsize, total_len, out_dim]
            return sequence

    def _train_forward(self, data, reshape=True):
        """
        training forward, predict demo_qoi_v from shot_num_min, and quest_qoi_v
        """
        basic_mask, index_pos, out_mask = self._get_matrices(data, mode="train", shot_num_min=self.shot_num_min)

        sequence = self._basic_forward(
            data=data,
            mode="train",
            index_pos=index_pos,
            basic_mask=basic_mask,
            shot_num_min=self.shot_num_min,
            need_weights=False,
        )
        sequence = sequence[:, out_mask, :]  # [batchsize, out_len, dim]
        if reshape:
            sequence = einops.rearrange(
                sequence, "b (x qoi_len) dim -> b x qoi_len dim", qoi_len=data["demo_qoi_v"].shape[-2]
            )  # [batchsize, x, qoi_len, dim], x depends on demo_num and shot_num_min
        return sequence

    def _test_forward(self, data, need_weights=False):
        """
        predict quest_qoi_v (with all demos)
        """
        # shot_num_min is useless for test mode
        basic_mask, index_pos, _ = self._get_matrices(data, mode="test", shot_num_min=0)
        if need_weights:
            sequence, weights = self._basic_forward(
                data=data, mode="test", index_pos=index_pos, basic_mask=basic_mask, shot_num_min=None, need_weights=True
            )
            quest_qoi_len = data["quest_qoi_mask"].shape[-1]
            sequence = sequence[:, None, -quest_qoi_len:, :]
            return sequence, weights
        else:
            sequence = self._basic_forward(
                data=data,
                mode="test",
                index_pos=index_pos,
                basic_mask=basic_mask,
                shot_num_min=None,
                need_weights=False,
            )
            quest_qoi_len = data["quest_qoi_mask"].shape[-1]
            sequence = sequence[:, None, -quest_qoi_len:, :]  # add batch dimension
            return sequence

    def forward(self, data, mode, **kwargs):
        if mode == "train":
            return self._train_forward(data, **kwargs)
        elif mode == "test":
            return self._test_forward(data, **kwargs)
        else:
            raise ValueError("mode should be 'train' or 'test'")
