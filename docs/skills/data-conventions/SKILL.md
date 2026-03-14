---
name: data-conventions
description: Data format conventions for datasets and plmodules
---

# Data Conventions

## Pytree as Data Structure

Data is passed around as pytrees — nested Python containers (lists, tuples, dicts). Every leaf must be a `torch.Tensor` or `np.ndarray` with **batch size as the leading dimension**.

- Use `torch.Tensor` for data that will be sent to devices (model inputs, labels).
- Use `np.ndarray` for data that won't be sent to devices (e.g., string descriptions use [NumPy string arrays](https://numpy.org/devdocs/user/basics.strings.html)).

## Dataset Item Format

The outermost pytree is typically a dict with these top-level keys:

- `description` — Human-readable explanation of the sample. Highly encouraged for debugging.
- `data` — Model input. Never include labels here.
- `label` *(optional)* — Ground truth.

### Operator Learning Example

```python
{
    "description": np.array(['...'] * batch_size, dtype=np.dtypes.StringDType()),
    "data": {
        "fx": torch.randn(batch_size, f_len, fx_dim),  # operator input function values
        "fy": torch.randn(batch_size, f_len, fy_dim),  # operator input function inputs
        "fm": torch.ones(batch_size, f_len, dtype=torch.bool),  # mask
        "gx": torch.randn(batch_size, g_len, gx_dim),  # operator output query points
        "gm": torch.ones(batch_size, g_len, dtype=torch.bool),
    },
    "label": torch.randn(batch_size, g_len, gy_dim),
}
```

`f` = operator input function, `g` = operator output function, `x` = function input, `y` = function output, `m` = mask. `gy` is excluded from `data` since it is the label.

### In-Context Operator Learning Example

```python
{
    "description": np.array(['...'] * batch_size, dtype=np.dtypes.StringDType()),
    "data": {
        "ex_fx": torch.randn(batch_size, ex_num, f_len, fx_dim),  # example input functions
        "ex_fy": torch.randn(batch_size, ex_num, f_len, fy_dim),
        "ex_fm": torch.ones(batch_size, ex_num, f_len, dtype=torch.bool),
        "ex_gx": torch.randn(batch_size, ex_num, g_len, gx_dim),
        "ex_gy": torch.randn(batch_size, ex_num, g_len, gy_dim),
        "ex_gm": torch.ones(batch_size, ex_num, g_len, dtype=torch.bool),
        "qn_fx": torch.randn(batch_size, qn_num, f_len, fx_dim),  # query input functions
        "qn_fy": torch.randn(batch_size, qn_num, f_len, fy_dim),
        "qn_fm": torch.ones(batch_size, qn_num, f_len, dtype=torch.bool),
        "qn_gx": torch.randn(batch_size, qn_num, g_len, gx_dim),
        "qn_gm": torch.ones(batch_size, qn_num, g_len, dtype=torch.bool),
    },
    "label": torch.randn(batch_size, qn_num, g_len, gy_dim),
}
```

`ex` = example, `qn` = question. `qn_gy` is excluded from `data` since it is the label.

## validation_step Return Format

`validation_step` in plmodules should return a pytree. Each leaf should be a `torch.Tensor` or `np.ndarray`, keeping the batch dimension (do not pool over batch).

Suggested top-level keys:

- `preds` — Model predictions (e.g., images or tensors).
- `errors` — Detailed prediction errors (e.g., per-sample error maps).
- `metrics` — Scalar-ish evaluation metrics with shape `(batch_size, ...)`. The flattened size of `...` should remain small.

The return value is passed directly as `outputs` to `on_validation_batch_end` in all callbacks.
