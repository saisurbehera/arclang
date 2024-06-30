import json
import numpy as np
import matplotlib.pyplot as plt
from arclang.image import Image, Piece
from matplotlib.colors import ListedColormap, BoundaryNorm

def display_matrices(matrices_dict):
    matrix_input = matrices_dict["input"].mask
    matrix_output = matrices_dict["output"].mask

    colors = [
        "#000000",  # black
        "#0074D9",  # blue
        "#FF4136",  # red
        "#2ECC40",  # green
        "#FFDC00",  # yellow
        "#AAAAAA",  # grey
        "#F012BE",  # fuchsia
        "#FF851B",  # orange
        "#7FDBFF",  # teal
        "#870C25",  # brown
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 10, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    cax1 = ax1.matshow(matrix_input, cmap=cmap, norm=norm)
    ax1.set_title("Input Matrix")

    cax2 = ax2.matshow(matrix_output, cmap=cmap, norm=norm)
    ax2.set_title("Output Matrix")

    fig.colorbar(
        cax1, ax=[ax1, ax2], ticks=np.arange(0, 10), orientation="vertical"
    ).ax.set_yticklabels(
        [
            "Symbol 0",
            "Symbol 1",
            "Symbol 2",
            "Symbol 3",
            "Symbol 4",
            "Symbol 5",
            "Symbol 6",
            "Symbol 7",
            "Symbol 8",
            "Symbol 9",
        ]
    )

    plt.show()

def display_matrix(matrix):
    colors = [
        "#000000",  # black
        "#0074D9",  # blue
        "#FF4136",  # red
        "#2ECC40",  # green
        "#FFDC00",  # yellow
        "#AAAAAA",  # grey
        "#F012BE",  # fuchsia
        "#FF851B",  # orange
        "#7FDBFF",  # teal
        "#870C25",  # brown
    ]
    cmap = ListedColormap(colors)
    bounds = np.arange(-0.5, 10, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cax = ax.matshow(matrix.mask, cmap=cmap, norm=norm)
    ax.set_title("Matrix")

    fig.colorbar(
        cax, ax=ax, ticks=np.arange(0, 10), orientation="vertical"
    ).ax.set_yticklabels(
        [
            "Symbol 0",
            "Symbol 1",
            "Symbol 2",
            "Symbol 3",
            "Symbol 4",
            "Symbol 5",
            "Symbol 6",
            "Symbol 7",
            "Symbol 8",
            "Symbol 9",
        ]
    )

    plt.show()


def display_matrix_term(matrix):
    # ANSI color codes
    colors = [
        "\033[40m",  # black
        "\033[44m",  # blue
        "\033[41m",  # red
        "\033[42m",  # green
        "\033[43m",  # yellow
        "\033[47m",  # grey
        "\033[45m",  # magenta
        "\033[48;5;208m",  # orange
        "\033[46m",  # cyan
        "\033[48;5;52m",  # brown
    ]
    reset = "\033[0m"

    print("Matrix:")
    for row in matrix.mask:
        for value in row:
            color = colors[value % len(colors)]
            print(f"{color}  {reset}", end="")
        print()
    print()


def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

def json_to_images(data):
    images = []
    for dataset in data:
        for item in dataset['train']:
            input_image = Image(mask=item['input'])
            output_image = Image(mask=item['output'])
            images.append((input_image, output_image))
    return images


def analyze_matrix_sizes(images):
    input_sizes = []
    output_sizes = []
    deltas = []

    for input_image, output_image in images:
        input_size = input_image.mask.shape
        output_size = output_image.mask.shape

        input_sizes.append(input_size)
        output_sizes.append(output_size)

        delta = (output_size[0] - input_size[0], output_size[1] - input_size[1])
        deltas.append(delta)

    input_flat_sizes = [x[0] * x[1] for x in input_sizes]
    output_flat_sizes = [x[0] * x[1] for x in output_sizes]
    delta_flat_sizes = [x[0] * x[1] for x in deltas]

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes[0].hist(input_flat_sizes, bins=20, color='blue', alpha=0.7)
    axes[0].set_title('Distribution of Input Matrix Sizes')
    axes[0].set_xlabel('Size')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(output_flat_sizes, bins=20, color='green', alpha=0.7)
    axes[1].set_title('Distribution of Output Matrix Sizes')
    axes[1].set_xlabel('Size')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(delta_flat_sizes, bins=20, color='red', alpha=0.7)
    axes[2].set_title('Delta between Output and Input Matrix Sizes')
    axes[2].set_xlabel('Delta Size')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
