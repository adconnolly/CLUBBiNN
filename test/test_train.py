"""
Unit tests training module utilities.

Test EarlyStopper and Trainer classes and saving functionality using pytest.
"""

import os
import tempfile
import json

import numpy as np
import pytest
import torch

from subgrid_parameterization.train.train import EarlyStopper, Trainer
from subgrid_parameterization.train.save import save_model, get_git_info


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


class TestSave:
    """Tests for code to save TorchScript models."""

    def test_get_git_info(self):
        """Tests we can get info - running inside repo so should return git data."""
        info = get_git_info()
        assert isinstance(info, dict)
        assert "branch" in info
        assert "commit" in info

    def test_save_model(self):
        model = DummyModel(2, 1)
        input_example = torch.randn(3, 2)
        input_vars = [{"name": "x1", "units": "none"}, {"name": "x2", "units": "none"}]
        output_vars = [{"name": "y", "units": "none"}]
        metrics = {"loss": 0.1, "R2": 0.9}
        train_config = {"epochs": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = "test_model"
            save_model(
                model,
                save_dir=tmpdir,
                filename=filename,
                input_example=input_example,
                input_vars=input_vars,
                output_vars=output_vars,
                metrics=metrics,
                train_config=train_config,
            )
            pt_path = os.path.join(tmpdir, f"{filename}.pt")
            meta_path = os.path.join(tmpdir, f"{filename}_metadata.json")

            # Check files were generated
            assert os.path.isfile(pt_path)
            assert os.path.isfile(meta_path)

            # Check model can be loaded from file and run with expected input shape to
            # generate expected output shape.
            model = torch.jit.load(pt_path)
            model.eval()
            input_tensor = torch.ones(3, 2)
            with torch.no_grad():
                output = model(input_tensor)
            assert output.shape == (3, 1)

            # Check metadata in json file is correct
            with open(meta_path, "r") as f:
                meta = json.load(f)
            assert meta["model_class"] == "DummyModel"
            assert meta["input_vars"] == input_vars
            assert meta["output_vars"] == output_vars
            assert meta["metrics"] == metrics
            assert meta["config"] == train_config
