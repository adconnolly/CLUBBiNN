"""Unit tests for EarlyStopper and Trainer classes using pytest."""

import os
import tempfile

import numpy as np
import pytest
import torch

from subgrid_parameterization.train.train import EarlyStopper, Trainer


class DummyDataset(torch.utils.data.Dataset):
    """Simple dataset for testing Trainer."""

    def __init__(self, n_samples=20, in_dim=4, out_dim=2):
        self.x = torch.randn(n_samples, in_dim)
        self.y = torch.randn(n_samples, out_dim)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DummyModel(torch.nn.Module):
    """Minimal model for Trainer tests."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def trainer_setup():
    """Create dummy parameters for everything needed to test the training loop."""
    config = {"batch_size": 4, "lr": 0.01, "wd": 0.0, "epochs": 3, "patience": 2}
    device = torch.device("cpu")
    lossweights = np.ones(2)
    trainer = Trainer(config, device, lossweights)

    in_dim = 4
    out_dim = 2
    model = DummyModel(in_dim, out_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )

    train_ds = DummyDataset(20, 4, 2)
    valid_ds = DummyDataset(10, 4, 2)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"]
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config["batch_size"]
    )

    return {
        "config": config,
        "device": device,
        "lossweights": lossweights,
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
    }


class TestEarlyStopper:
    """Tests for EarlyStopper early stopping logic."""

    @pytest.mark.parametrize(
        "losses,patience,min_delta,should_stop",
        [
            # patience=2, min_delta=0.1
            ([1.0, 0.9, 1.01, 1.0, 1.1], 2, 0.1, True),  # Should stop at last
            ([1.0, 0.9, 1.01, 1.0, 1.1], 3, 0.1, False),  # Longer patience
            ([1.0, 0.9, 1.01, 1.0, 1.1], 2, 0.2, False),  # Larger delta
            ([1.0, 0.95, 0.94, 0.93], 2, 0.01, False),  # Always improving
            ([1.0, 1.0, 1.0, 1.0], 2, 0.0, False),  # No improvement never stops
            ([1.0, 0.9, 1.01, 1.0, 0.8, 1.1, 0.85], 2, 0.1, False),  # stopper reset
        ],
    )
    def test_early_stop(self, losses, patience, min_delta, should_stop):
        """Test EarlyStopper triggers after patience exceeded."""
        stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        stops = [stopper.early_stop(loss) for loss in losses]
        assert all(s is False for s in stops[:-1]) if not should_stop else True
        assert stops[-1] is should_stop


class TestTrainer:
    """Tests for Trainer training loop."""

    def test_train_loop_runs(self, trainer_setup):
        """Test Trainer.train_loop runs and returns trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_name = os.path.join(tmpdir, "testmodel")
            trained_model = trainer_setup["trainer"].train_loop(
                trainer_setup["model"],
                trainer_setup["optimizer"],
                trainer_setup["train_loader"],
                trainer_setup["valid_loader"],
                save_name,
            )
            assert isinstance(trained_model, DummyModel)

    def test_model_parameters_change_after_training(self, trainer_setup):
        """Test that model parameters change after training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_name = os.path.join(tmpdir, "testmodel")
            initial_params = [p.clone() for p in trainer_setup["model"].parameters()]
            trained_model = trainer_setup["trainer"].train_loop(
                trainer_setup["model"],
                trainer_setup["optimizer"],
                trainer_setup["train_loader"],
                trainer_setup["valid_loader"],
                save_name,
            )
            changed = any(
                not torch.equal(p0, p1)
                for p0, p1 in zip(initial_params, trained_model.parameters())
            )
            assert changed

    def test_train_loop_zero_epochs(self, trainer_setup):
        """Test Trainer.train_loop with zero epochs does not train."""
        trainer_setup["trainer"].config["epochs"] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            save_name = os.path.join(tmpdir, "testmodel")
            initial_params = [p.clone() for p in trainer_setup["model"].parameters()]
            with pytest.warns(UserWarning, match="No training occurred"):
                trained_model = trainer_setup["trainer"].train_loop(
                    trainer_setup["model"],
                    trainer_setup["optimizer"],
                    trainer_setup["train_loader"],
                    trainer_setup["valid_loader"],
                    save_name,
                )
            for p1, p2 in zip(initial_params, trained_model.parameters()):
                torch.testing.assert_close(p1, p2)

    def test_train_loop_mismatched_lossweights_dimension(self):
        """Test Trainer raises error with mismatched lossweights shape."""
        config = {"batch_size": 4, "lr": 0.01, "wd": 0.0, "epochs": 1, "patience": 1}
        device = torch.device("cpu")
        lossweights = np.ones((2, 2))  # Not 1D

        with pytest.raises(RuntimeError, match="lossweights must be a 1D array"):
            Trainer(config, device, lossweights)
