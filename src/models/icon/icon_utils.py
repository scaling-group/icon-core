#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import numpy as np
import torch


def build_diag_block(cond_len, qoi_kv_len, qoi_k_len):
    diag_block = np.zeros((cond_len + qoi_kv_len + qoi_k_len, cond_len + qoi_kv_len + qoi_k_len), dtype=bool)
    diag_block[:, :cond_len] = 1
    diag_block[cond_len : cond_len + qoi_kv_len, cond_len : cond_len + qoi_kv_len] = 1
    diag_block[cond_len + qoi_kv_len :, cond_len + qoi_kv_len :] = np.eye(qoi_k_len, dtype=bool)
    return torch.tensor(diag_block, dtype=torch.bool)


def build_bool_sequence(demo_num, mode, shot_num_min):
    """
    shot_num_min only works for train mode, useless for test mode, which predicts the quest with all demos
    """
    if mode == "train":
        cond_list = [True] * demo_num + [True]
        qoi_kv_list = [True] * demo_num + [False]
        qoi_k_list = [i >= shot_num_min for i in range(demo_num)] + [True]
    elif mode == "test":
        cond_list = [True] * demo_num + [True]
        qoi_kv_list = [True] * demo_num + [False]
        qoi_k_list = [False] * demo_num + [True]
    else:
        raise ValueError(f"not supported mode: {mode}")
    return cond_list, qoi_kv_list, qoi_k_list


def build_basic_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list):
    assert len(cond_len_list) == len(qoi_kv_len_list) == len(qoi_k_len_list), "Length of lists should be equal"
    num = len(cond_len_list)
    mask_size = sum([cond_len_list[i] + qoi_kv_len_list[i] + qoi_k_len_list[i] for i in range(num)])
    mask = np.zeros((mask_size, mask_size), dtype=bool)

    for i in range(num):
        for j in range(i + 1):
            cond_len_i = cond_len_list[i]
            qoi_kv_len_i = qoi_kv_len_list[i]
            qoi_k_len_i = qoi_k_len_list[i]
            cursor_i = sum([cond_len_list[k] + qoi_kv_len_list[k] + qoi_k_len_list[k] for k in range(i)])
            block_size_i = cond_len_i + qoi_kv_len_i + qoi_k_len_i

            cond_len_j = cond_len_list[j]
            qoi_kv_len_j = qoi_kv_len_list[j]
            qoi_k_len_j = qoi_k_len_list[j]
            cursor_j = sum([cond_len_list[k] + qoi_kv_len_list[k] + qoi_k_len_list[k] for k in range(j)])
            block_size_j = cond_len_j + qoi_kv_len_j + qoi_k_len_j

            if i == j:
                mask[cursor_i : cursor_i + block_size_i, cursor_j : cursor_j + block_size_j] = build_diag_block(
                    cond_len_i, qoi_kv_len_i, qoi_k_len_i
                )
            else:
                mask[cursor_i : cursor_i + block_size_i, cursor_j : cursor_j + cond_len_j + qoi_kv_len_j] = True

    return torch.tensor(mask, dtype=torch.bool)


def build_index_integer(cond_len_list, qoi_kv_len_list, qoi_k_len_list):
    index = []
    num = len(cond_len_list)
    for i in range(num):
        cond_len = cond_len_list[i]
        qoi_kv_len = qoi_kv_len_list[i]
        qoi_k_len = qoi_k_len_list[i]
        index += [i * 3] * cond_len + [i * 3 + 1] * qoi_kv_len + [i * 3 + 2] * qoi_k_len
    return torch.tensor(index)


def build_out_mask(cond_len_list, qoi_kv_len_list, qoi_k_len_list, num_range):
    assert len(cond_len_list) == len(qoi_kv_len_list) == len(qoi_k_len_list), "Length of lists should be equal"
    num = len(cond_len_list)
    out_mask_size = sum([cond_len_list[i] + qoi_kv_len_list[i] + qoi_k_len_list[i] for i in range(num)])
    out_mask = np.zeros((out_mask_size), dtype=bool)

    begin, end = num_range

    cursor = 0
    for i in range(num):
        cond_len = cond_len_list[i]
        qoi_kv_len = qoi_kv_len_list[i]
        qoi_k_len = qoi_k_len_list[i]
        pair_size = cond_len + qoi_kv_len + qoi_k_len
        if i >= begin and i < end:
            out_mask[cursor + cond_len + qoi_kv_len : cursor + pair_size] = 1
        cursor += pair_size

    return torch.tensor(out_mask, dtype=torch.bool)


def build_data_sequence(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list, data_quest_qoi_v=None):
    demo_cond = torch.cat([data["demo_cond_k"], data["demo_cond_v"]], dim=-1)
    demo_qoi_kv = torch.cat([data["demo_qoi_k"], data["demo_qoi_v"]], dim=-1)
    demo_qoi_k = torch.nn.functional.pad(data["demo_qoi_k"], (0, data["demo_qoi_v"].shape[-1]))

    if data_quest_qoi_v is None:
        data_quest_qoi_v = torch.zeros(
            (
                data["quest_qoi_k"].shape[0],
                data["quest_qoi_k"].shape[1],
                data["quest_qoi_k"].shape[2],
                data["demo_qoi_v"].shape[-1],
            ),
            device=data["demo_qoi_v"].device,
        )
    quest_cond = torch.cat([data["quest_cond_k"], data["quest_cond_v"]], dim=-1)
    quest_qoi_kv = torch.cat([data["quest_qoi_k"], data_quest_qoi_v], dim=-1)
    quest_qoi_k = torch.nn.functional.pad(data["quest_qoi_k"], (0, data_quest_qoi_v.shape[-1]))

    demo_num = data["demo_cond_k"].shape[1]

    sequence = []
    for i in range(demo_num):
        if cond_bool_list[i]:
            sequence.append(demo_cond[:, i])
        if qoi_kv_bool_list[i]:
            sequence.append(demo_qoi_kv[:, i])
        if qoi_k_bool_list[i]:
            sequence.append(demo_qoi_k[:, i])
    if cond_bool_list[-1]:
        sequence.append(quest_cond[:, 0])
    if qoi_kv_bool_list[-1]:
        sequence.append(quest_qoi_kv[:, 0])
    if qoi_k_bool_list[-1]:
        sequence.append(quest_qoi_k[:, 0])
    sequence = torch.cat(sequence, dim=1)

    return sequence


def build_data_mask(data, cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list):
    demo_num = data["demo_cond_k"].shape[1]

    mask = []
    for i in range(demo_num):
        if cond_bool_list[i]:
            mask.append(data["demo_cond_mask"][:, i])
        if qoi_kv_bool_list[i]:
            mask.append(data["demo_qoi_mask"][:, i])
        if qoi_k_bool_list[i]:
            mask.append(data["demo_qoi_mask"][:, i])
    if cond_bool_list[-1]:
        mask.append(data["quest_cond_mask"][:, 0])
    if qoi_kv_bool_list[-1]:
        mask.append(data["quest_qoi_mask"][:, 0])
    if qoi_k_bool_list[-1]:
        mask.append(data["quest_qoi_mask"][:, 0])

    mask = torch.cat(mask, dim=1).bool()
    # careful: here one means attention, zero means no attention, so reverse it when using
    return mask


def build_matrices(data_shape, mode, shot_num_min, returns=("mask", "index", "out")):
    """
    data_shape: dict, the shape of the data, no batch size
    careful: here one means attention, zero means no attention, so reverse it when using
    """
    demo_num = data_shape["demo_cond_k"][0]
    demo_cond_len = data_shape["demo_cond_k"][1]
    demo_qoi_len = data_shape["demo_qoi_k"][1]
    quest_cond_len = data_shape["quest_cond_k"][1]
    quest_qoi_len = data_shape["quest_qoi_k"][1]

    cond_bool_list, qoi_kv_bool_list, qoi_k_bool_list = build_bool_sequence(demo_num, mode, shot_num_min)

    cond_len_list_raw = [demo_cond_len] * demo_num + [quest_cond_len]
    qoi_kv_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]
    qoi_k_len_list_raw = [demo_qoi_len] * demo_num + [quest_qoi_len]

    cond_len_list = [i * j for i, j in zip(cond_bool_list, cond_len_list_raw, strict=False)]
    qoi_kv_len_list = [i * j for i, j in zip(qoi_kv_bool_list, qoi_kv_len_list_raw, strict=False)]
    qoi_k_len_list = [i * j for i, j in zip(qoi_k_bool_list, qoi_k_len_list_raw, strict=False)]

    return_list = []
    if "mask" in returns:
        basic_mask = build_basic_mask(
            cond_len_list=cond_len_list, qoi_kv_len_list=qoi_kv_len_list, qoi_k_len_list=qoi_k_len_list
        )
        return_list.append(basic_mask)
    if "index" in returns:
        index_pos = build_index_integer(
            cond_len_list=cond_len_list, qoi_kv_len_list=qoi_kv_len_list, qoi_k_len_list=qoi_k_len_list
        )
        return_list.append(index_pos)
    if "out" in returns:
        out_mask = build_out_mask(
            cond_len_list=cond_len_list,
            qoi_kv_len_list=qoi_kv_len_list,
            qoi_k_len_list=qoi_k_len_list,
            num_range=(shot_num_min, demo_num + 1),
        )
        return_list.append(out_mask)
    if "len" in returns:
        return_list.extend([cond_len_list, qoi_kv_len_list, qoi_k_len_list])

    return tuple(return_list)
