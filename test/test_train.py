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

from pathlib import Path

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
        info = get_git_info(Path(__file__).parent)
        assert isinstance(info, dict)
        assert "branch" in info
        assert "commit" in info

    def test_valid_vars(self):
        """Test that valid variables pass through unchanged."""
        input_vars = [
            {
                "name": "temp",
                "desc": "Temperature field normalised by surface value",
                "shape": [10, 5],
            },
            {
                "name": "pressure",
                "desc": "Pressure field normalised by surface value",
                "shape": [10, 5],
            },
        ]
        from subgrid_parameterization.train.save import validate_var_list

        result = validate_var_list(input_vars, "input")
        assert result == input_vars

    def test_missing_required_keys(self):
        """Test that missing required keys trigger warnings."""
        invalid_vars = [
            {"desc": "Temperature normalised by sfc", "shape": [10, 5]},
            {"name": "pressure"},
        ]
        from subgrid_parameterization.train.save import validate_var_list

        with pytest.warns(UserWarning) as warning_info:
            result = validate_var_list(invalid_vars, "input")

        # Check that warnings were raised
        assert len(warning_info) == 2
        assert "Missing required keys" in str(warning_info[0].message)
        assert "Missing required keys" in str(warning_info[1].message)

        # Check that variables with missing keys are still included
        assert len(result) == 2
        assert result[0]["desc"] == "Temperature normalised by sfc"
        assert result[1]["name"] == "pressure"

    def test_non_dict_variables(self):
        """Test that non-dict variables raise ValueError."""
        invalid_vars = [
            {"name": "temp", "desc": "Temperature normalised by surface value"},
            "not_a_dict",  # This should raise an error
            {"name": "pressure", "desc": "Pressure normalised by surface value"},
        ]
        from subgrid_parameterization.train.save import validate_var_list

        with pytest.raises(
            ValueError, match="input variable at index 1 must be a dictionary"
        ):
            validate_var_list(invalid_vars, "input")

    def test_save_model_with_extended_metadata(self):
        """Test saving model with metadata fields."""
        model = DummyModel(2, 1)
        input_example = torch.randn(3, 2)
        input_vars = [
            {"name": "x1", "desc": "First input feature", "shape": [3, 2]},
            {"name": "x2", "desc": "Second input feature", "shape": [3, 2]},
        ]
        output_vars = [{"name": "y", "desc": "Output prediction", "shape": [3, 1]}]

        # Extended metrics
        metrics = {
            "loss": 0.1,
            "R2": 0.9,
            "train_R2": 0.92,
            "val_R2": 0.88,
            "steps": 1000,
            "early_stop_time": 50,
        }

        # Extended config
        train_config = {
            "train_dataset": "/full/path/to/train_data.nc",
            "val_dataset": "/full/path/to/val_data.nc",
            "Hscale": 100.0,
            "LMax": 1000.0,
        }

        other_notes = "This is a test run with extended metadata"

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
                other_notes=other_notes,
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
            assert meta["other_notes"] == other_notes
