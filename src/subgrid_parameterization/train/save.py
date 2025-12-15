import os
import json
import torch
import subprocess


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
    except Exception:
        return {}


def save_model(
    model,
    save_dir=".",
    filename="model_scripted",
    input_example=None,
    input_vars=None,
    output_vars=None,
    metrics=None,
    train_config=None,
):
    """
    Save a TorchScript version of the model and a metadata JSON file.

    The mode will be saved to ``filename.pt``, the metadata will be saved to

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
        [{"name": ..., "units": ..., "shape": ...}, ...].
    output_vars : list of dict, optional
        List of dictionaries describing output variables.
        [{"name": ..., "units": ..., "shape": ...}, ...].
    metrics : dict, optional
        Dictionary of performance metrics.
        {"loss": ..., "R2": ...}.
    train_config : dict, optional
        Training configuration dictionary.
        {"input_dataset": ..., etc.}

    Returns
    -------
    None
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

    # Gather and save metadata
    metadata = {
        "model_class": type(model).__name__,
        "input_vars": input_vars,
        "output_vars": output_vars,
        "metrics": metrics,
        "config": train_config,
        "git": get_git_info(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
