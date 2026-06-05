"""
下载 MNIST 数据集并保存为 PNG 图片。
使用 scikit-learn 的 fetch_openml 获取数据。
"""
import os

import cv2
import numpy as np
from sklearn.datasets import fetch_openml

DATA_DIR = "data/mnist"


def save_images(images: np.ndarray, labels: np.ndarray, output_dir: str, max_per_label: int = 100):
    """按标签保存 PNG 图片。"""
    os.makedirs(output_dir, exist_ok=True)

    count_per_label = {i: 0 for i in range(10)}
    saved = 0

    for idx, (image_vec, label) in enumerate(zip(images, labels)):
        label = int(label)
        if count_per_label[label] >= max_per_label:
            continue

        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # reshape 784 -> 28x28
        image = image_vec.reshape(28, 28).astype(np.uint8)
        filepath = os.path.join(label_dir, f"{idx:05d}.png")
        cv2.imwrite(filepath, image)

        count_per_label[label] += 1
        saved += 1

    print(f"  ✓ Saved {saved} images to {output_dir}")
    for i in range(10):
        print(f"    Label {i}: {count_per_label[i]} images")


def main():
    print("=" * 60)
    print("📥 Loading MNIST dataset via fetch_openml")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("  Downloading (this may take a minute)...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
    X = mnist.data.astype(np.float32)  # [70000, 784] range [0, 255]
    y = mnist.target.astype(np.int64)  # [70000]

    print(f"  Total samples: {len(X)}")

    # 训练集前 60000，测试集后 10000
    train_images, train_labels = X[:60000], y[:60000]
    test_images, test_labels = X[60000:], y[60000:]

    print("\n📂 Saving training images ...")
    save_images(train_images, train_labels, os.path.join(DATA_DIR, "train"), max_per_label=100)

    print("\n📂 Saving test images ...")
    save_images(test_images, test_labels, os.path.join(DATA_DIR, "test"), max_per_label=100)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
