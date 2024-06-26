import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import List, Tuple, Union, Callable

from arclang.image import Image, Point

Point = namedtuple("Point", ["x", "y"])


def visualize_center(original: Image, centered: Image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(original.mask, cmap="viridis")
    ax1.set_title(f"Original ({original.w}x{original.h})")
    ax1.axis("off")

    if centered is not None:
        full_size = np.zeros((original.h, original.w))
        y_start, x_start = centered.y - original.y, centered.x - original.x
        full_size[y_start : y_start + centered.h, x_start : x_start + centered.w] = (
            centered.mask
        )

        ax2.imshow(full_size, cmap="viridis")
        ax2.set_title(f"Centered ({centered.w}x{centered.h})")
    else:
        ax2.text(0.5, 0.5, "Centering failed", ha="center", va="center")
        ax2.set_title("Centered (Error)")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_compose(a: Image, b: Image, result: Image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(a.mask, cmap="viridis")
    ax1.set_title(f"Image A ({a.w}x{a.h})")
    ax1.axis("off")

    ax2.imshow(b.mask, cmap="viridis")
    ax2.set_title(f"Image B ({b.w}x{b.h})")
    ax2.axis("off")

    if result is not None:
        ax3.imshow(result.mask, cmap="viridis")
        ax3.set_title(f"Composed ({result.w}x{result.h})")
    else:
        ax3.text(0.5, 0.5, "Composition failed", ha="center", va="center")
        ax3.set_title("Composed (Error)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_count(img: Image, result: Image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img.mask, cmap="viridis")
    ax1.set_title(f"Input ({img.w}x{img.h})")
    ax1.axis("off")

    if result is not None:
        ax2.imshow(result.mask, cmap="viridis")
        ax2.set_title(f"Count Result ({result.w}x{result.h})")
    else:
        ax2.text(0.5, 0.5, "Count failed", ha="center", va="center")
        ax2.set_title("Count (Error)")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_cut(img: Image, mask: Image, result: List[Image]):
    n_results = len(result) if result else 0
    fig, axs = plt.subplots(1, 3 + n_results, figsize=(5 * (3 + n_results), 5))

    axs[0].imshow(img.mask, cmap="viridis")
    axs[0].set_title(f"Input ({img.w}x{img.h})")
    axs[0].axis("off")

    axs[1].imshow(mask.mask, cmap="viridis")
    axs[1].set_title(f"Mask ({mask.w}x{mask.h})")
    axs[1].axis("off")

    if result:
        for i, r in enumerate(result):
            axs[i + 2].imshow(r.mask, cmap="viridis")
            axs[i + 2].set_title(f"Cut {i+1} ({r.w}x{r.h})")
            axs[i + 2].axis("off")
    else:
        axs[2].text(0.5, 0.5, "Cut failed", ha="center", va="center")
        axs[2].set_title("Cut (Error)")
        axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_extend(img: Image, room: Image, result: Image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(img.mask, cmap="viridis")
    ax1.set_title(f"Input ({img.w}x{img.h})")
    ax1.axis("off")

    ax2.imshow(room.mask, cmap="viridis")
    ax2.set_title(f"Room ({room.w}x{room.h})")
    ax2.axis("off")

    if result is not None:
        ax3.imshow(result.mask, cmap="viridis")
        ax3.set_title(f"Extended ({result.w}x{result.h})")
    else:
        ax3.text(0.5, 0.5, "Extend failed", ha="center", va="center")
        ax3.set_title("Extended (Error)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_get_regular(img: Image, result: Image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img.mask, cmap="viridis")
    ax1.set_title(f"Input ({img.w}x{img.h})")
    ax1.axis("off")

    if result is not None:
        ax2.imshow(result.mask, cmap="viridis")
        ax2.set_title(f"Regular ({result.w}x{result.h})")
    else:
        ax2.text(0.5, 0.5, "Get regular failed", ha="center", va="center")
        ax2.set_title("Regular (Error)")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_mirror(a: Image, b: Image, result: Image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(a.mask, cmap="viridis")
    ax1.set_title(f"Input A ({a.w}x{a.h})")
    ax1.axis("off")

    ax2.imshow(b.mask, cmap="viridis")
    ax2.set_title(f"Input B ({b.w}x{b.h})")
    ax2.axis("off")

    if result is not None:
        ax3.imshow(result.mask, cmap="viridis")
        ax3.set_title(f"Mirrored ({result.w}x{result.h})")
    else:
        ax3.text(0.5, 0.5, "Mirror failed", ha="center", va="center")
        ax3.set_title("Mirrored (Error)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_outer_product(a: Image, b: Image, result: Image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(a.mask, cmap="viridis")
    ax1.set_title(f"Input A ({a.w}x{a.h})")
    ax1.axis("off")

    ax2.imshow(b.mask, cmap="viridis")
    ax2.set_title(f"Input B ({b.w}x{b.h})")
    ax2.axis("off")

    if result is not None:
        ax3.imshow(result.mask, cmap="viridis")
        ax3.set_title(f"Outer Product ({result.w}x{result.h})")
    else:
        ax3.text(0.5, 0.5, "Outer product failed", ha="center", va="center")
        ax3.set_title("Outer Product (Error)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_repeat(a: Image, b: Image, result: Image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(a.mask, cmap="viridis")
    ax1.set_title(f"Input A ({a.w}x{a.h})")
    ax1.axis("off")

    ax2.imshow(b.mask, cmap="viridis")
    ax2.set_title(f"Input B ({b.w}x{b.h})")
    ax2.axis("off")

    if result is not None:
        ax3.imshow(result.mask, cmap="viridis")
        ax3.set_title(f"Repeated ({result.w}x{result.h})")
    else:
        ax3.text(0.5, 0.5, "Repeat failed", ha="center", va="center")
        ax3.set_title("Repeated (Error)")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_rigid(img: Image, result: Image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img.mask, cmap="viridis")
    ax1.set_title(f"Input ({img.w}x{img.h})")
    ax1.axis("off")

    if result is not None:
        ax2.imshow(result.mask, cmap="viridis")
        ax2.set_title(f"Rigid Transformed ({result.w}x{result.h})")
    else:
        ax2.text(0.5, 0.5, "Rigid transform failed", ha="center", va="center")
        ax2.set_title("Rigid Transformed (Error)")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_split_cols(img: Image, result: List[Image]):
    n_results = len(result) if result else 0
    fig, axs = plt.subplots(1, 2 + n_results, figsize=(5 * (2 + n_results), 5))

    axs[0].imshow(img.mask, cmap="viridis")
    axs[0].set_title(f"Input ({img.w}x{img.h})")
    axs[0].axis("off")

    if result:
        for i, r in enumerate(result):
            axs[i + 1].imshow(r.mask, cmap="viridis")
            axs[i + 1].set_title(f"Split {i+1} ({r.w}x{r.h})")
            axs[i + 1].axis("off")
    else:
        axs[1].text(0.5, 0.5, "Split cols failed", ha="center", va="center")
        axs[1].set_title("Split Cols (Error)")
        axs[1].axis("off")

    plt.tight_layout()
    plt.show()
