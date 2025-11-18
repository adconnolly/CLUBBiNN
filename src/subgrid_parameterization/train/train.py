"""Training utilities for use in code."""

import warnings
import numpy as np
import torch


class EarlyStopper:
    """Class to trigger early stopping if loss threshold is reached in training."""

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, val_loss):
        """
        Check whether to trigger early stopping.

        Based on loss not decreasing below best yet for a certain number of steps set
        by the patience attribute.
        """
        if val_loss < self.min_validation_loss:
            self.min_validation_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    """Class setting up a training loop structure for use in code."""

    def __init__(self, config, device, lossweights):
        self.config = config
        """Typed Dict of {batch_size, lr, wd, epochs, patience}"""
        self.train_loss = []
        self.test_loss = []
        self.device = device
        self.lossweights = torch.from_numpy(lossweights).to(self.device)

        if len(self.lossweights.shape) != 1:
            raise RuntimeError("lossweights must be a 1D array")

    def train_loop(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        save_name: str,
        verbose=False,
    ) -> torch.nn.Module:
        """
        Run training loop for given model using provided data loaders and optimizer.

        Periodically saves best performing (lowest loss) model before loading and
        returning at the end.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to train.
        optimizer : torch.optim.Optimizer
            The optimizer used for training.
        train_loader : torch.utils.data.DataLoader
            DataLoader for training data.
        valid_loader : torch.utils.data.DataLoader
            DataLoader for validation data.
        save_name : str
            The filename for saving the best models to.
        verbose : bool
            Whether to provide verbose logging output.

        Returns
        -------
        torch.nn.Module
            The trained model with the best validation performance.
        """
        # Ensure model is on device
        model.to(self.device)

        # Set up early stopper for training
        early_stopper = EarlyStopper(patience=self.config["patience"], min_delta=0)

        # Set up loss criterion
        criterion = torch.nn.MSELoss()
        lossmin = np.inf

        for epoch in range(self.config["epochs"]):
            train_samples = 0
            train_running_loss = 0.0
            valid_samples = 0
            valid_running_loss = 0.0

            model.train()
            for data in train_loader:
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)

                ## zero the parameter gradients
                optimizer.zero_grad()

                output = model(x_data)  ## Takes in Q, outputs \hat{S}
                loss = criterion(output * self.lossweights, y_data * self.lossweights)
                loss.backward()
                optimizer.step()

                ## Store loss values
                train_running_loss += loss.detach() * x_data.shape[0]
                train_samples += x_data.shape[0]
            train_running_loss /= train_samples

            model.eval()
            for data in valid_loader:
                x_data, y_data = data
                x_data = x_data.to(self.device)
                y_data = y_data.to(self.device)

                optimizer.zero_grad()
                output = model(x_data)  ## Takes in Q, outputs \hat{S}
                val_loss = criterion(output, y_data)

                ## Store loss values
                valid_running_loss += val_loss.detach() * x_data.shape[0]
                valid_samples += x_data.shape[0]
            valid_running_loss /= valid_samples

            early_stop = early_stopper.early_stop(valid_running_loss)
            if early_stop:
                print(
                    f"Early stopping epoch: {epoch - early_stopper.patience}"  # +/- 1?
                )
                break
            if valid_running_loss < lossmin:
                lossmin = valid_running_loss
                torch.save(model.state_dict(), save_name + "_net.pt")
                torch.save(optimizer.state_dict(), save_name + "_optim.pt")

            # Push loss values for each epoch to wandb
            log_dic = {}
            log_dic["epoch"] = epoch
            log_dic["training_loss"] = (
                (train_running_loss / train_samples).cpu().numpy()
            )
            log_dic["valid_loss"] = (valid_running_loss / valid_samples).cpu().numpy()
            # wandb.log(log_dic)
            self.train_loss.append((train_running_loss / train_samples).cpu().numpy())
            self.test_loss.append((valid_running_loss / valid_samples).cpu().numpy())

            if verbose:
                print(
                    f"{log_dic['epoch']:03d} "
                    f"{log_dic['training_loss']:.3e} "
                    f"{log_dic['valid_loss']:.3e}"
                )

        # If no training occurred, save initial model so load does not fail
        if self.config["epochs"] <= 0:
            warnings.warn(
                "Trainer.train_loop: No training occurred (epochs=0). Saving initial model state."
            )
            torch.save(model.state_dict(), save_name + "_net.pt")
        # Load the best performing (lowest loss) model and return
        model.load_state_dict(
            torch.load(save_name + "_net.pt", weights_only=True)
        )  # ,map_location=device),strict=False)

        return model
