#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import io

import matplotlib.pyplot as plt
import wandb
from matplotlib.figure import Figure
from PIL import Image


def fig_to_wandb(fig: Figure | Image.Image, cfg=None):
    """
    Converts a Matplotlib figure or a PIL image to a wandb.Image object
    Parameters:
    - fig: figure to be converted.
    Returns:
    - wandb.Image object
    """
    if cfg is None:
        cfg = {}

    if isinstance(fig, Figure):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        wandb_image = wandb.Image(image, **cfg)
        buf.close()
    else:  # already a PIL image
        wandb_image = wandb.Image(fig, **cfg)
    return wandb_image


def merge_images(figs_2d, spacing=0):
    """
    Merges a 2D list of Matplotlib figures or PIL images into a single PIL image arranged in a grid.

    Parameters:
      - figs_2d: 2D list of Matplotlib figures or PIL images to be merged.
      - spacing: Space between images in pixels.

    Returns:
      - Merged PIL image.
    """
    # List to store merged images of each row.
    row_images = []
    total_width = 0
    max_row_height = 0

    # Process each row of figures/images.
    for figs_row in figs_2d:
        imgs = []
        bufs = []
        # Convert each item to PIL image if it's a Matplotlib figure
        for item in figs_row:
            if isinstance(item, Figure):
                buf = io.BytesIO()
                bufs.append(buf)
                item.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                img = Image.open(buf)
                imgs.append(img)
            else:  # assume it's already a PIL image
                imgs.append(item)

        # Compute the size required for this row.
        row_width = sum(img.width for img in imgs) + spacing * (len(imgs) - 1)
        row_height = max(img.height for img in imgs)
        total_width = max(total_width, row_width)
        max_row_height = max(max_row_height, row_height)

        # Create a new image for the row and paste the images side by side.
        row_image = Image.new("RGB", (row_width, row_height))
        x_offset = 0
        for img in imgs:
            row_image.paste(img, (x_offset, 0))
            x_offset += img.width + spacing
        row_images.append(row_image)

        # Close buffers but keep images open for now
        for buf in bufs:
            buf.close()
        # Close the Matplotlib figures if they exist.
        for item in figs_row:
            if isinstance(item, Figure):
                plt.close(item)

    # Determine the total height for the final merged image.
    total_height = sum(row_img.height for row_img in row_images) + spacing * (len(row_images) - 1)

    # Create the final merged image and paste each row image.
    merged_img = Image.new("RGB", (total_width, total_height))
    y_offset = 0
    for row_img in row_images:
        merged_img.paste(row_img, (0, y_offset))
        y_offset += row_img.height + spacing

    # Don't close row images as they are now part of the merged image
    return merged_img
