import einops
import torch.nn as nn


class Conv2d(nn.Module):
    """
    a wrapper of nn.Conv2d, with support of arbitrary leading dimensions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [..., *leading_dims, in_channels, h, w]
        Returns:
            output: [..., *leading_dims, out_channels, h, w]
        """
        leading_dims = x.shape[:-3]

        # Flatten leading dimensions to make it compatible with Conv2d
        x_flat = einops.rearrange(x, "... c h w -> (...) c h w")

        # Apply convolution
        output_flat = self.conv(x_flat)

        # Restore leading dimensions
        if leading_dims:
            # Build the rearrange pattern dynamically
            leading_pattern = " ".join([f"dim{i}" for i in range(len(leading_dims))])
            pattern = f"({leading_pattern}) c h w -> {leading_pattern} c h w"
            output = einops.rearrange(
                output_flat, pattern, **{f"dim{i}": leading_dims[i] for i in range(len(leading_dims))}
            )
        else:
            output = output_flat

        return output
