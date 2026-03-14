#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch


def patchify(x, patch_num):
    """
    x: (bs, c, ph, pw)
    return: (bs, p * p, c * h * w)
    """
    bs, c, hp, wp = x.shape
    h = hp // patch_num
    w = wp // patch_num
    p = patch_num

    patches = x.view(bs, c, p, h, p, w).permute(0, 2, 4, 1, 3, 5)  # (bs, c, p, h, p, w) -> (bs, p, p, c, h, w)
    patches = patches.reshape((bs, p * p, c * h * w))
    return patches


def depatchify(patches, patch_num, c, h, w):
    """
    patches: (bs, p * p, c * h * w)
    return: (bs, c, ph, pw)
    """
    bs = patches.shape[0]
    p = patch_num

    patches = patches.view(bs, p, p, c, h, w).permute(0, 3, 1, 4, 2, 5)  # (bs, p, p, c, h, w) -> (bs, c, p, h, p, w)
    x = patches.reshape((bs, c, p * h, p * w))
    return x


def build_alternating_block_lowtri_mask(block_num, block_size1, block_size2):
    """
    lower triangular block mask
    for example: block_num = 3, block_size1 = 2, block_size2 = 1, ninf = false
    1,1,0,0,0,0,0,0,0
    1,1,0,0,0,0,0,0,0
    1,1,1,0,0,0,0,0,0
    1,1,1,1,1,0,0,0,0
    1,1,1,1,1,0,0,0,0
    1,1,1,1,1,1,0,0,0
    1,1,1,1,1,1,1,1,0
    1,1,1,1,1,1,1,1,0
    1,1,1,1,1,1,1,1,1
    block_num: the number of blocks
    block_size1,2: the size of each block 1,2
    return: (block_num * (block_size1 + block_size2), block_num * * (block_size1 + block_size2))
    """
    # Create a standard lower triangular matrix
    # base_mask = torch.tril(torch.ones(block_num, block_num))
    # the inner iteration on the base mask if either base_mask_inner or base_mask_inner_tail
    # when the base_mask_inner is full 1,
    # while the base_mask_inner_tail is like below, both with size (block_size1+block_size2) x (block_size1+block_size2)
    # 1,1,0
    # 1,1,0
    # 1,1,1
    # for example: block_size1 = 2, block_size2 = 1, ninf = false
    base_mask_inner = torch.ones((block_size1 + block_size2, block_size1 + block_size2))
    base_mask_inner_tail = torch.ones((block_size1 + block_size2, block_size1 + block_size2))
    # setting the leading block_size1 row, and col after block_size1 to 0
    base_mask_inner_tail[:block_size1, block_size1:] = 0

    # Expand each element into alternating blocks of size block_size1 and block_size2
    expanded_mask = torch.zeros(block_num, block_num, block_size1 + block_size2, block_size1 + block_size2)
    for i in range(block_num):
        for j in range(i):
            expanded_mask[i, j] = base_mask_inner
        expanded_mask[i, i] = base_mask_inner_tail
        # block_size = block_size1 if i % 2 == 0 else block_size2
        # expanded_mask[i, :i+1, :block_size, :block_size] = base_mask[i, :i+1, None, None]
    # print(expanded_mask)
    permuted_mask = expanded_mask.permute(0, 2, 1, 3)

    # Reshape to get the final block mask
    final_mask = permuted_mask.reshape(block_num * (block_size1 + block_size2), block_num * (block_size1 + block_size2))

    return final_mask
