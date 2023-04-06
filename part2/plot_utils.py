import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, titles, dpi=50):

    n_row, n_col = len(images), len(images[0])
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row), facecolor='white', dpi=dpi)
    ax = ax.flatten()

    for i in range(n_row):
        for j in range(n_col):
            k = i * n_col + j
            ax[k].imshow(images[i][j], cmap='gray')
            ax[k].set_title(titles[i][j])
            ax[k].axis('off')

    plt.tight_layout()
    plt.show()


def plot_overlay_edges(edges, titles, bg_image, dpi=50, colour=(100, 255, 100)):

    n_row, n_col = len(edges), len(edges[0])
    fig, ax = plt.subplots(n_row, n_col, figsize=(4*n_col, 4*n_row), facecolor='white', dpi=dpi)
    ax = ax.flatten()

    for i in range(n_row):
        for j in range(n_col):
            overlay = np.repeat(np.expand_dims(bg_image, -1), 3, axis=-1).astype(np.uint8)
            overlay[np.where(edges[i][i] == 255)] = colour
            k = i * n_col + j
            ax[k].imshow(overlay, cmap='gray')
            ax[k].set_title(titles[i][j])
            ax[k].axis('off')

    plt.tight_layout()
    plt.show()