from matplotlib import pyplot as plt


def show_images(images, cols=5, figsize=(12, 10)):
    n = len(images)
    rows = (n // cols) + 1

    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray" if img.mode == "L" else None)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
