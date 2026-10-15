import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from core.autoencoder import Autoencoder
from training.train import Trainer
from training.dataset import create_dataset


def visualize_reconstruction(
    model,
    test_data,
    image_size=8,
    output_path="reconstruction.png"
):
    """
    Save a comparison of original and reconstructed images.
    """
    model.eval()

    # Select a few test images.
    count = min(5, len(test_data))

    if count == 0:
        print("No test images available for visualization.")
        return

    images = torch.as_tensor(
        test_data[:count],
        dtype=torch.float32
    )

    # Generate reconstructions without tracking gradients.
    with torch.no_grad():
        reconstructed = model(images)

    originals = images.numpy()
    reconstructed = reconstructed.numpy()

    fig, axes = plt.subplots(
        2,
        count,
        figsize=(2 * count, 4)
    )

    if count == 1:
        axes = axes.reshape(2, 1)

    for i in range(count):
        axes[0, i].imshow(
            originals[i].reshape(image_size, image_size),
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(
            reconstructed[i].reshape(image_size, image_size),
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"\nVisualization saved to: {output_path}")


def main():
    # Reproducibility.
    np.random.seed(42)
    torch.manual_seed(42)

    # Dataset configuration.
    image_size = 8
    input_size = image_size * image_size

    # 1. Create the dataset.
    train_data, test_data = create_dataset(
        num_samples=1000,
        image_size=image_size,
        test_ratio=0.2
    )

    print(f"Training images: {len(train_data)}")
    print(f"Test images: {len(test_data)}")
    print(f"Input dimensions: {input_size}")

    # 2. Build the Autoencoder.
    model = Autoencoder(
        input_size=input_size,
        latent_size=16
    )

    print("\nModel architecture:")
    print(model)

    # 3. Train the model.
    trainer = Trainer(
        model=model,
        learning_rate=0.001,
        epochs=100,
        batch_size=16
    )

    print("\nStarting training")

    history = trainer.train(
        train_data=train_data,
        test_data=test_data
    )

    # 4. Report final losses
    print("\n✅ Training Complete")

    print(
        f"Final training loss: "
        f"{history['train_loss'][-1]:.6f}"
    )

    if history["test_loss"]:
        print(
            f"Final test loss: "
            f"{history['test_loss'][-1]:.6f}"
        )

    # 5. Save reconstruction visualization.
    visualize_reconstruction(
        model=model,
        test_data=test_data,
        image_size=image_size
    )

if __name__ == "__main__":
    main()
