import torch
import copy
import torch.multiprocessing as mp

class EnKFOptimizerGradFree:
    def __init__(self, model, lr=1e-3, sigma=0.1, k=10, gamma=1e-3, max_iterations=10, debug_mode=False):
        self.model = model
        self.lr = lr
        self.sigma = sigma
        self.k = k
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.parameters = list(model.parameters())
        self.theta = torch.cat([p.data.view(-1) for p in self.parameters])  #Flattened parameters
        self.shapes = [p.shape for p in self.parameters]  #For keeping track of original shapes
        self.cumulative_sizes = [0] + list(torch.cumsum(torch.tensor([p.numel() for p in self.parameters]), dim=0))
        self.debug_mode = debug_mode
        self.particles = None

    def flatten_parameters(self, parameters):
        '''
        The weights from all the layers will be considered as a single vector
        '''
        return torch.cat([p.data.view(-1) for p in parameters])

    def unflatten_parameters(self, flat_params):
        '''
        Here, we regain the shape to so that we can use them to evaluate the model
        '''
        params_list = []
        start = 0
        for shape in self.shapes:
            num_elements = torch.prod(torch.tensor(shape))
            params_list.append(flat_params[start:start + num_elements].view(shape))
            start += num_elements
        return params_list
    

    def step(self, F, obs):
        for iteration in range(self.max_iterations):
            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Started")

            # Step [1] Draw K Particles
            self.Omega = torch.randn((self.theta.numel(), self.k)) * self.sigma  # Draw particles
            particles = self.theta.unsqueeze(1) + self.Omega  # Add the noise to the current parameter estimate
            self.particles = particles  # This is for retrieving

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Drawing {self.k} Particles completed")

            # Step [2] Evaluate the forward model using theta mean
            current_params_unflattened = self.unflatten_parameters(self.theta)
            with torch.no_grad():
                F_current = F(current_params_unflattened)

            # Ensure F_current and F_perturbed are 4D tensors
            batch_size, channels, height, width = F_current.size()  # [batch_size, channels, height, width]
            n_out = channels * height * width  # Total number of features per sample
            Q = torch.zeros(batch_size, n_out, self.k)  # [batch_size, n_out, k]

            for i in range(self.k):
                perturbed_params = particles[:, i]
                perturbed_params_unflattened = self.unflatten_parameters(perturbed_params)

                with torch.no_grad():
                    F_perturbed = F(perturbed_params_unflattened)

                # Compute the difference and flatten the spatial dimensions
                F_perturbed_flat = F_perturbed.view(batch_size, -1)  # Flatten to [batch_size, n_out]
                F_current_flat = F_current.view(batch_size, -1)  # Flatten to [batch_size, n_out]
                Q[:, :, i] = F_perturbed_flat - F_current_flat


            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : forward model evaluation complete")

            # Step [3] Construct the Hessian Matrix Hj = Qj(transpose) x Qj + Γ
            Q_vec = Q.view(batch_size * n_out, self.k)  # Reshape Q to [batch_size * n_out, k]
            H_j = Q_vec.T @ Q_vec + self.gamma * torch.eye(self.k)
            H_inv = torch.inverse(H_j)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : Hj and Hj inverse completed")

            # Step [4] Calculate the Gradient of loss function with respect to the current parameters
            gradient = self.misfit_gradient(F, self.theta, obs)
            gradient = gradient.view(-1, 1)  # Ensure it's a column vector

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : gradient calculation completed")

            # Step [5] Update the parameters
            adjustment = H_inv @ Q_vec.T  # Shape [k, batch_size * n_out]

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : adjustment calculated")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.Omega = self.Omega.to(device)
            adjustment = adjustment.to(device)
            gradient = gradient.to(device)

            # Perform the optimized operations on the GPU
            intermediate_result = adjustment @ gradient  # Shape [k, 1]
            update = self.Omega @ intermediate_result  # Shape [n, 1]

            update = update.view(-1)  # Reshape to [n]

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} :  update reshaped")

            self.theta -= self.lr * update  # Now both are [n]

            self.update_model_parameters(self.theta)

            if self.debug_mode:
                print(f"iteration {iteration + 1} / {self.max_iterations} : parameter update completed")
            


    def update_model_parameters(self, flat_params):
        idx = 0
        for param in self.model.parameters():
            #param.grad = None
            num_elements = param.numel()
            param.data.copy_(flat_params[idx:idx + num_elements].reshape(param.shape))
            idx += num_elements

    def misfit_gradient(self, F, thetha, d_obs):
        #Forward pass to get model outputs
        t = F(self.unflatten_parameters(thetha))
        
        #compute simple residuals
        residuals = t - d_obs

        return residuals.view(-1, 1)


    def simple_line_search(self, F, update, initial_lr, obs, reduction_factor=0.5, max_reductions=5):
        lr = initial_lr
        current_params_unflattened = self.unflatten_parameters(self.theta)
        
        # Compute the initial predictions and loss directly
        current_predictions = F(current_params_unflattened)
        current_loss = torch.mean((current_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

        for _ in range(max_reductions):
            new_theta = self.theta - lr * update
            new_predictions = F(self.unflatten_parameters(new_theta))
            new_loss = torch.mean((new_predictions - obs) ** 2).item()  # Compute MSE and convert to scalar

            if new_loss < current_loss:
                return lr
            
            lr -= lr*reduction_factor

        return lr
