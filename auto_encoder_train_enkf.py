import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optimiser.enkf import EnKFOptimizerGradFree
from model.autoencoder import DnCNN
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelTrainer():
    def __init__(self, model, lr=0.5, sigma=0.001, k=25, gamma=1e-1, max_iterations=1):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimiser = EnKFOptimizerGradFree(model, lr, sigma, k, gamma, max_iterations, debug_mode=False)

    def load_data(self, set_standardize=False, test_size=0.2, val_size=0.2, subset_size=1000):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Take a subset of the dataset
        X_train, y_train = mnist_train.data[:subset_size].unsqueeze(1).float() / 255.0, mnist_train.targets[:subset_size]
        X_test, y_test = mnist_test.data[:subset_size].unsqueeze(1).float() / 255.0, mnist_test.targets[:subset_size]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

        # Adding noise to the images
        noise_factor = 0.5
        X_train_noisy = X_train + noise_factor * torch.randn(*X_train.shape)
        X_val_noisy = X_val + noise_factor * torch.randn(*X_val.shape)
        X_test_noisy = X_test + noise_factor * torch.randn(*X_test.shape)

        # Clipping to maintain [0, 1] range
        X_train_noisy = torch.clip(X_train_noisy, 0., 1.)
        X_val_noisy = torch.clip(X_val_noisy, 0., 1.)
        X_test_noisy = torch.clip(X_test_noisy, 0., 1.)

        self.X_train, self.X_train_noisy = X_train, X_train_noisy
        self.X_val, self.X_val_noisy = X_val, X_val_noisy
        self.X_test, self.X_test_noisy = X_test, X_test_noisy

        self.__convert_data_to_tensor()

    def __convert_data_to_tensor(self):
        # Convert to tensors
        self.X_train = self.X_train.to(torch.float32)
        self.X_train_noisy = self.X_train_noisy.to(torch.float32)
        self.X_val = self.X_val.to(torch.float32)
        self.X_val_noisy = self.X_val_noisy.to(torch.float32)
        self.X_test = self.X_test.to(torch.float32)
        self.X_test_noisy = self.X_test_noisy.to(torch.float32)

    def F(self, parameters):
            with torch.no_grad():
                for original_param, new_param in zip(self.model.parameters(), parameters):
                    original_param.data.copy_(new_param.data)

                # Perform the forward pass with the adjusted parameters
                output = self.model(self.X_train_noisy)
            return output

    def train(self, num_epochs=100, is_plot_graph=1):
        train_losses = []
        val_losses = []

        print("TRAINING STARTED ...")

        for epoch in range(num_epochs):
            self.optimiser.step(F=self.F, obs=self.X_train)

            # Evaluate on training and validation set
            with torch.no_grad():
                train_output = self.model(self.X_train_noisy)
                train_loss = self.loss_function(train_output, self.X_train)
                val_output = self.model(self.X_val_noisy)
                val_loss = self.loss_function(val_output, self.X_val)
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}')

        if is_plot_graph:
            self.plot_train_graph(train_losses, val_losses)

        self.train_loss = train_losses
        self.val_loss = val_losses

    def plot_train_graph(self, train_losses, val_losses):
        # Plot training and validation loss
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            test_output = self.model(self.X_test_noisy)
            test_loss = self.loss_function(test_output, self.X_test)
        print(f'Test Loss: {test_loss.item()}')

    def save_model(self, filename=None):
        if filename is None:
            filename = f'model_enkf.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model, save_path)
        print(f'Complete model saved to {save_path}')


    def visualize_denoising(self, num_images=10):
        self.model.eval()
        with torch.no_grad():
            # Fetch a subset of test images
            X_test_noisy = self.X_test_noisy[:num_images]
            X_test = self.X_test[:num_images]

            # Denoise the images
            denoised_images = self.model(X_test_noisy)

            # Plot the noisy and denoised images
            fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(10, num_images * 2))

            for i in range(num_images):
                # Original image
                axes[i, 0].imshow(X_test[i].squeeze(0).cpu().numpy(), cmap='gray')
                axes[i, 0].set_title("Original")
                axes[i, 0].axis('off')

                # Noisy image
                axes[i, 1].imshow(X_test_noisy[i].squeeze(0).cpu().numpy(), cmap='gray')
                axes[i, 1].set_title("Noisy")
                axes[i, 1].axis('off')

                # Denoised image
                axes[i, 2].imshow(denoised_images[i].squeeze(0).cpu().numpy(), cmap='gray')
                axes[i, 2].set_title("Denoised")
                axes[i, 2].axis('off')

            plt.tight_layout()
            plt.show()


# Training Script
if __name__ == "__main__":
    model = DnCNN()
    trainer = ModelTrainer(model=model, lr=0.5, sigma=0.001, k=100, gamma=1e-1, max_iterations=1)
    trainer.load_data()
    trainer.train(is_plot_graph=1)
    trainer.evaluate()
    trainer.visualize_denoising(num_images=10)
    #trainer.save_model('autoencoder_enkf.pth')