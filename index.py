import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optimiser.enkf import EnKFOptimizerGradFree
from model.dnn import DNN
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ModelTrainer():
    def __init__(self, model, lr=0.5, sigma=0.001, k=10, gamma=1e-1, max_iterations=1):
        self.model = model
        self.loss_function = nn.MSELoss()
        self.optimiser = EnKFOptimizerGradFree(model, lr, sigma, k, gamma, max_iterations=1, debug_mode=False)


    def load_data(self, data, target, set_standardize=False, test_size=0.2, val_size=0.2):
        # Split data into training and temporary set
        X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=test_size + val_size, random_state=42)

        # Split temporary set into validation and test sets
        val_size_adjusted = val_size / (test_size + val_size)  # Adjust validation size for the reduced dataset
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        if set_standardize:
            self.standardize_data()

        self.__convert_data_to_tensor()

    def standardize_data(self):
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def __convert_data_to_tensor(self):
        # Convert to tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

    def F(self, parameters):
        with torch.no_grad():
            for original_param, new_param in zip(self.model.parameters(), parameters):
                original_param.data.copy_(new_param.data)

            # Perform the forward pass with the adjusted parameters
            output = self.model(self.X_train)
        return output

    def train(self, num_epochs=50, is_plot_graph=0):
        train_losses = []
        val_losses = []

        print("TRAINING STARTED ...")

        for epoch in range(num_epochs):
            self.optimiser.step(F=self.F, obs=self.y_train)

            # Evaluate on training and validation set
            with torch.no_grad():
                train_output = self.model(self.X_train)
                train_loss = self.loss_function(train_output, self.y_train)
                val_output = self.model(self.X_val)
                val_loss = self.loss_function(val_output, self.y_val)
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
            test_output = self.model(self.X_test)
            test_loss = self.loss_function(test_output, self.y_test)
        print(f'Test Loss: {test_loss.item()}')

    def save_model(self, filename=None):
        if filename is None:
            filename = f'model_enkf.pth'
        save_path = os.path.join('./saved_models', filename)
        torch.save(self.model, save_path)
        print(f'Complete model saved to {save_path}')

    def get_ensemble_particles(self):
        return [self.optimiser.unflatten_parameters(particle) for particle in self.optimiser.particles.T]

    def plot_ensemble_particles_distribution(self):
        particles = self.get_ensemble_particles()
        flattened_particles = [np.concatenate([p.detach().cpu().numpy().flatten() for p in particle]) for particle in particles]

        num_particles = len(flattened_particles)
        fig = make_subplots(rows=num_particles, cols=1, subplot_titles=[f'Particle {i+1} Distribution' for i in range(num_particles)])

        for i, particle in enumerate(flattened_particles):
            fig.add_trace(
                go.Histogram(x=particle, nbinsx=50, name=f'Particle {i+1}'),
                row=i + 1, col=1
            )

        fig.update_layout(height=300 * num_particles, width=800, title_text="Particle Distributions", showlegend=False)
        fig.show()

    def visualize_denoising(self, num_images=10):
        self.model.eval()
        with torch.no_grad():
            # Select a batch of test images
            X_test_noisy, X_test = next(iter(self.test_loader))
            X_test_noisy = X_test_noisy[:num_images]
            X_test = X_test[:num_images]
            
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


# Dataset
data = pd.read_csv('./dataset/oscillatory_data_small.csv')
X = data[[col for col in data.columns if 'Theta' in col]].values
y = data[[col for col in data.columns if 'F_Theta' in col]].values

print(y.shape)

model_train = ModelTrainer(model=DNN(input_size=X.shape[1], output_size=y.shape[1]))
model_train.load_data(data=X, target=y)
model_train.train(is_plot_graph=1)
model_train.evaluate()

model_train.save_model('model_enkf.pth')
