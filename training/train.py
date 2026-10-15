import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Trainer:
    """
    Handles training and evaluation of the Autoencoder.

    PyTorch manages gradient calculation and parameter updates.
    The Trainer manages the training loop and reconstruction loss.
    """

    def __init__(
        self,
        model,
        learning_rate=0.001,
        epochs=100,
        batch_size=16
    ):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size

        # Reconstruction loss.
        self.criterion = nn.MSELoss()

        # Optimizer updates the model parameters.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.history = {
            "train_loss": [],
            "test_loss": []
        }

    def _prepare_data(self, data):
        """
        Convert NumPy data into float32 PyTorch tensors.
        """
        return torch.as_tensor(
            np.asarray(data),
            dtype=torch.float32
        )

    def train(self, train_data, test_data=None):
        """
        Train the Autoencoder.

        Args:
            train_data: Training images as flattened arrays.
            test_data: Optional held-out test images.

        Returns:
            Training history.
        """
        train_tensor = self._prepare_data(train_data)

        test_tensor = (
            self._prepare_data(test_data)
            if test_data is not None
            else None
        )

        dataset = torch.utils.data.TensorDataset(
            train_tensor
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        for epoch in range(1, self.epochs + 1):

            # Enable training behavior.
            self.model.train()

            total_loss = 0.0
            total_samples = 0

            for (batch,) in loader:

                # Clear gradients from the previous step.
                self.optimizer.zero_grad()

                # Forward pass.
                reconstruction = self.model(batch)

                # Compare reconstruction with original input.
                loss = self.criterion(
                    reconstruction,
                    batch
                )

                # Calculate gradients automatically.
                loss.backward()

                # Update model parameters.
                self.optimizer.step()

                total_loss += (
                    loss.item() * batch.size(0)
                )
                total_samples += batch.size(0)

            train_loss = total_loss / total_samples

            self.history["train_loss"].append(
                train_loss
            )

            # Evaluate without updating parameters.
            test_loss = None

            if test_tensor is not None and len(test_tensor) > 0:

                self.model.eval()

                with torch.no_grad():
                    reconstruction = self.model(
                        test_tensor
                    )

                    test_loss = self.criterion(
                        reconstruction,
                        test_tensor
                    ).item()

                self.history["test_loss"].append(
                    test_loss
                )

            # Print progress.
            if (
                epoch == 1
                or epoch % 10 == 0
                or epoch == self.epochs
            ):
                message = (
                    f"Epoch {epoch}/{self.epochs}"
                    f" | Train Loss: {train_loss:.6f}"
                )

                if test_loss is not None:
                    message += (
                        f" | Test Loss: {test_loss:.6f}"
                    )

                print(message)

        return self.history
