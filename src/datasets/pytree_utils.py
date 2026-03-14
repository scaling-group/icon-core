#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

from collections.abc import Sequence

import einops
import numpy as np
import optree
import torch
from frozendict import frozendict
from optree import PyTree
from rich.tree import Tree

# See docs: https://optree.readthedocs.io/en/latest/pytree.html


def to_hashable_pytree(batch: PyTree) -> PyTree:
    """
    Return a new pytree with list -> tuple and dict -> frozendict recursively.
    """
    if isinstance(batch, dict):
        return frozendict({k: to_hashable_pytree(v) for k, v in batch.items()})
    elif isinstance(batch, list):
        return tuple(to_hashable_pytree(item) for item in batch)
    elif isinstance(batch, np.ndarray | torch.Tensor):
        if batch.ndim == 0:
            return batch.item()
        else:
            return tuple(to_hashable_pytree(item) for item in batch)
    else:
        return batch


def to_numpy(batch: PyTree) -> PyTree:
    """
    Return a new pytree with all torch.Tensors converted to numpy arrays.
    """
    return optree.tree_map(lambda leaf: einops.asnumpy(leaf.float()), batch)


def to_tensor_all(batch: PyTree, *args, **kwargs) -> PyTree:
    """
    Return a new pytree with all numpy arrays converted to torch.Tensors.
    **Warning:** This function is usually unnecessary,
    as our PyTrees are expected to contain `np.ndarray`s with non-numeric objects.
    """
    return optree.tree_map(
        lambda leaf: torch.tensor(leaf, *args, **kwargs) if isinstance(leaf, np.ndarray) else leaf,
        batch,
    )


def is_np_numeric(leaf):
    return isinstance(leaf, np.ndarray) and (
        np.issubdtype(leaf.dtype, np.number)
        or np.issubdtype(leaf.dtype, np.bool_)
        or np.issubdtype(leaf.dtype, np.bool)
    )


def to_tensor_numeric(batch: PyTree, *args, **kwargs) -> PyTree:
    """
    Return a new pytree with all numpy arrays converted to torch.Tensors, only for numeric batch
    """
    return optree.tree_map(
        lambda leaf: torch.tensor(leaf, *args, **kwargs) if is_np_numeric(leaf) else leaf,
        batch,
    )


to_tensor = to_tensor_numeric  # alias


def get_one_sample(batch: PyTree, bid: int, keep_dim=False) -> PyTree:
    """
    Return a new pytree with the bid-th sample of each leaf
    """

    def get_one_sample_leaf(leaf):
        if isinstance(leaf, np.ndarray | torch.Tensor):
            return leaf[bid : bid + 1] if keep_dim else leaf[bid]
        else:
            return leaf  # should not happen in general

    return optree.tree_map(get_one_sample_leaf, batch)


def get_slice_batch(batch: PyTree, bid_list: Sequence[int]) -> PyTree:
    """
    Return a new pytree with the sliced leaves
    """
    return optree.tree_map(
        lambda leaf: leaf[bid_list] if isinstance(leaf, np.ndarray | torch.Tensor) else leaf,
        batch,
    )


def concat(batches: Sequence[PyTree], dim=0) -> PyTree:
    """
    Concatenate a list of pytrees
    """
    return optree.tree_map(
        lambda *leaves: np.concatenate(leaves, axis=dim)
        if isinstance(leaves[0], np.ndarray)
        else torch.cat(leaves, dim=dim),
        *batches,
    )


def get_shape(batch: PyTree, exclude_batch: bool = False) -> PyTree:
    """
    Return a new pytree with the shape of each leaf
    If exclude_batch is True, exclude the leading batch dimension from the shape.
    """

    def get_shape_leaf(leaf):
        if not isinstance(leaf, np.ndarray | torch.Tensor):
            return None  # this should not happen in general
        if exclude_batch:
            return leaf.shape[1:]
        else:
            return leaf.shape

    return optree.tree_map(get_shape_leaf, batch)


def truncate_seq(value: list, max_len: int) -> list:
    """
    Truncate the sequence to the first 2 and the last 2 elements
    """
    if len(value) <= max_len:
        return value
    else:
        half_len = max_len // 2
        return value[:half_len] + ["..."] + value[-half_len:]


def get_print_info_lv0(batch: PyTree) -> PyTree:
    """
    Return the print info of the batch, only get the first level of the batch
    """
    if "description" not in batch:
        description = "Warning: no description in the batch"
    else:
        description = [str(s) for s in batch["description"]]
        description = truncate_seq(description, max_len=4)
    return {"description": description, "keys": ", ".join(list(batch.keys()))}


def get_array_tensor_info(leaf: np.ndarray | torch.Tensor) -> str:
    try:
        if isinstance(leaf, np.ndarray):
            if np.issubdtype(leaf.dtype, np.floating):
                range_str = f"range={leaf.min():.3f}:{leaf.max():.3f}|{leaf.mean():.3f}~{leaf.std():.3f}"
            elif not np.issubdtype(leaf.dtype, np.complexfloating):  # should be integer or bool
                range_str = (
                    f"range={leaf.min()}:{leaf.max()}|{leaf.astype(float).mean():.3f}~{leaf.astype(float).std():.3f}"
                )
            else:
                range_str = ""
            return f"{type(leaf).__name__} {leaf.shape} {leaf.dtype} {range_str} "
        if isinstance(leaf, torch.Tensor):
            if torch.is_floating_point(leaf):
                range_str = f"range={leaf.min():.3f}:{leaf.max():.3f}|{leaf.mean():.3f}~{leaf.std():.3f}"
            elif not torch.is_complex(leaf):  # should be integer or bool
                range_str = f"range={leaf.min()}:{leaf.max()}|{leaf.float().mean():.3f}~{leaf.float().std():.3f}"
            else:
                range_str = ""
            return f"{type(leaf).__name__} {leaf.shape} {leaf.dtype} {range_str} {leaf.device}"

    except Exception as e:
        return f"Error when getting array/tensor info: {e}"


def get_print_info_lv1(batch: PyTree) -> PyTree:
    """
    Return the print info of the batch, return a pytree with the print info of each leaf
    """

    def get_print_info_leaf(leaf):
        if not isinstance(leaf, np.ndarray | torch.Tensor):
            return f"type={type(leaf)}, warning: non-tensor/array leaf"  # this should not happen in general
        if isinstance(leaf, np.ndarray) and np.issubdtype(leaf.dtype, np.dtypes.StringDType()):
            info_list = [str(s) for s in leaf]
            return truncate_seq(info_list, max_len=4)
        return get_array_tensor_info(leaf)

    return optree.tree_map(get_print_info_leaf, batch)


def get_print_info_lv2(batch: PyTree) -> PyTree:
    """
    Return the print info of the batch, return a pytree with the print info of each leaf
    """

    def get_print_info_leaf(leaf):
        if not isinstance(leaf, np.ndarray | torch.Tensor):
            return f"type={type(leaf)}, warning: non-tensor/array leaf"  # this should not happen in general
        if isinstance(leaf, np.ndarray) and np.issubdtype(leaf.dtype, np.dtypes.StringDType()):
            return [str(s) for s in leaf]
        return get_array_tensor_info(leaf)

    return optree.tree_map(get_print_info_leaf, batch)


def pytree_to_rich_tree(node, name="root"):
    """
    Convert a pytree to a Rich Tree object for pretty printing
    """
    if isinstance(node, dict):
        tree = Tree(f"[bold]{name}")
        for key, value in node.items():
            tree.add(pytree_to_rich_tree(value, str(key)))
        return tree
    elif isinstance(node, list | tuple):
        tree = Tree(f"[bold]{name}")
        for i, value in enumerate(node):
            tree.add(pytree_to_rich_tree(value, str(i)))
        return tree
    else:
        return Tree(f"{name}: [green]{node}")


def get_print_info(batch: PyTree, print_lv: int, info: str = "") -> Tree:
    """
    print the information of the batch, no newline in the end
    """
    # Create a root tree with the begin info
    doc = "-" * 20 + f"begin {info} " + "-" * 20

    # Add the appropriate content tree based on print level
    if print_lv == 0:
        rich_tree = pytree_to_rich_tree(get_print_info_lv0(batch), name=doc)
    elif print_lv == 1:
        rich_tree = pytree_to_rich_tree(get_print_info_lv1(batch), name=doc)
    elif print_lv == 2:
        rich_tree = pytree_to_rich_tree(get_print_info_lv2(batch), name=doc)
    else:
        raise ValueError(f"Unknown print_lv: {print_lv}")

    return rich_tree


def get_discription_list(batch: PyTree) -> list:
    """
    Return the description list of the batch
    """
    if "description" not in batch:
        return ["Warning: no description in the batch"]
    else:
        return [str(s) for s in batch["description"]]


if __name__ == "__main__":
    import typing

    batch = {
        "description": np.array(["test", "test2"], dtype=np.dtypes.StringDType()),
        "data": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "label": torch.tensor([3, 4], dtype=torch.int32),
    }

    from rich import print

    print(get_print_info(batch, print_lv=0))
    print(get_print_info(batch, print_lv=1))
    print(get_print_info(batch, print_lv=2))
    hashable_batch = to_hashable_pytree(batch)
    print(hashable_batch)
    print(isinstance(hashable_batch, typing.Hashable), hash(hashable_batch))

    batch["data"] += 1
    hashable_batch = to_hashable_pytree(batch)
    print(hashable_batch)
    print(isinstance(hashable_batch, typing.Hashable), hash(hashable_batch))

    batch["label"] += 1
    hashable_batch = to_hashable_pytree(batch)
    print(hashable_batch)
    print(isinstance(hashable_batch, typing.Hashable), hash(hashable_batch))
