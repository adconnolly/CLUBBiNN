import os
import json
import torch
import subprocess
import warnings


def get_git_info(repo_dir="."):
    """
    Get the current git branch and commit hash for the repository at repo_dir.

    Parameters
    ----------
    repo_dir : str, optional
        Path to the git repository (default is current directory).

    Returns
    -------
    dict
        Dictionary with 'branch' and 'commit' keys, or empty dict if unavailable.
    """
    try:
        branch = (
            subprocess.check_output(
                ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        commit = (
            subprocess.check_output(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return {"branch": branch, "commit": commit}
    except subprocess.CalledProcessError:
        return {}


def validate_var_list(var_list, var_type):
    """
    Validate and optionally normalize a list of variable dictionaries.

    Parameters
    ----------
    var_list : list of dict
        List of variable dictionaries to validate
    var_type : str
        Type of variables ('input' or 'output') for error messages

    Returns
    -------
    list of dict
        Validated and optionally normalized variable list
    """
    if not var_list:
        return var_list

    validated_vars = []
    required_keys = {"name", "desc"}

    for i, var in enumerate(var_list):
        if not isinstance(var, dict):
            raise ValueError(
                f"{var_type} variable at index {i} must be a dictionary, got {type(var)}"
            )

        # Check for missing required keys
        missing_keys = required_keys - set(var.keys())
        if missing_keys:
            warnings.warn(
                f"Missing required keys in {var_type} variable at index {i}: {missing_keys}. "
                f"Current keys: {list(var.keys())}. "
            )

        # Create new dict with required and optional keys
        validated_var = {"name": var.get("name"), "desc": var.get("desc")}

        # Add shape if present
        if "shape" in var:
            validated_var["shape"] = var.get("shape")

        validated_vars.append(validated_var)

    return validated_vars


def save_model(
    model,
    save_dir=".",
    filename="model_scripted",
    input_example=None,
    input_vars=None,
    output_vars=None,
    metrics=None,
    train_config=None,
    other_notes=None,
):
    """
    Save a TorchScript version of the model and a metadata JSON file.

    The model will be saved to ``filename.pt``, the metadata will be saved to
    ``filename_metadata.json``.


    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model to be scripted and saved.
    save_dir : str, optional
        Directory to save files to (defaults to current directory).
    filename : str, optional
        Filename for the outputs (default 'model_scripted').
    input_example : torch.Tensor, optional
        Example input tensor for tracing the model. Default of None in which case
        scripting will be used.
    input_vars : list of dict, optional
        List of dictionaries describing input variables.
        [{"name": ..., "desc": ..., "shape": ...}, ...].
    output_vars : list of dict, optional
        List of dictionaries describing output variables.
        [{"name": ..., "desc": ..., "shape": ...}, ...].
    metrics : dict, optional
        Dictionary of performance metrics.
        {"loss": ..., "R2": ..., "train_R2": ..., "val_R2": ..., "steps": ...,
         "early_stop_time": ...}.
    train_config : dict, optional
        Training configuration dictionary.
        {"input_dataset": ..., etc.}
    other_notes : str, optional
        Additional notes about the training run.

    Returns
    -------
    None

    Examples
    --------
    >>> # Model to be saved and example input tensor
    >>> model = MyModel()
    >>> input_example = torch.randn(10, 5)
    >>> input_vars = [
    ...     {"name": "temperature", "desc": "Normalized temperature field", "shape": [10, 5]},
    ...     {"name": "pressure", "desc": "Normalized pressure field", "shape": [10, 5]}
    ... ]
    >>> output_vars = [{"name": "prediction", "desc": "Model output", "shape": [10, 1]}]
    >>> metrics = {
    ...     "loss": 0.05,
    ...     "R2": 0.95,
    ...     "train_R2": 0.96,
    ...     "val_R2": 0.94,
    ...     "steps": 5000,
    ...     "early_stop_time": 200
    ... }
    >>> train_config = {
    ...     "train_dataset": "/data/train_dataset.nc",
    ...     "val_dataset": "/data/val_dataset.nc",
    ...     "Hscale": 100.0,
    ...     "LMax": 1000.0
    ... }
    >>> other_notes = "Training completed successfully with early stopping"
    >>>
    >>> save_model(
    ...     model=model,
    ...     save_dir="./saved_models",
    ...     filename="my_trained_model",
    ...     input_example=input_example,
    ...     input_vars=input_vars,
    ...     output_vars=output_vars,
    ...     metrics=metrics,
    ...     train_config=train_config,
    ...     other_notes=other_notes
    ... )
    """
    os.makedirs(save_dir, exist_ok=True)
    scripted_path = os.path.join(save_dir, f"{filename}.pt")
    metadata_path = os.path.join(save_dir, f"{filename}_metadata.json")

    # Save the model to TorchScript
    if input_example is not None:
        scripted_model = torch.jit.trace(model, input_example)
    else:
        scripted_model = torch.jit.script(model)
    scripted_model.save(scripted_path)

    # Validate input and output variables
    if input_vars is not None:
        input_vars = validate_var_list(input_vars, "input")
    if output_vars is not None:
        output_vars = validate_var_list(output_vars, "output")

    # Gather and save metadata
    metadata = {
        "model_class": type(model).__name__,
        "input_vars": input_vars,
        "output_vars": output_vars,
        "metrics": metrics,
        "config": train_config,
        "git": get_git_info(),
    }

    # Add other_notes if provided
    if other_notes is not None:
        metadata["other_notes"] = other_notes

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
