import torch
import numpy as np
import itertools

""" Fully connected artificial neural network """


## Construct a convolutional block
def make_layer(
    in_features: int,
    out_features: int,
    activation=torch.nn.ReLU,
    batch_norm=False,
    bias=False,
) -> list:
    """
    Packs ANN layer and optionally ReLU/BatchNorm2d
    layers in a list
    """
    layer = [torch.nn.Linear(in_features, out_features, bias=bias)]
    if activation is not None:
        layer.append(activation())
    if batch_norm:
        layer.append(torch.nn.BatchNorm2d(out_features))
    return layer


class ANN(torch.nn.Module):
    def __init__(self, N, activation=torch.nn.ReLU):
        """
        Packs sequence of artificial neural network layers in a list.
        """
        super().__init__()
        ops = []
        # Following recipe for itertools.pairwise
        NA, NB = itertools.tee(N)
        next(NB, None)
        n_iter = zip(NA, NB)
        # Create all but the last linear layer and activation
        for n_a, n_b in itertools.islice(n_iter, len(N) - 2):
            ops.extend(
                make_layer(
                    in_features=n_a,
                    out_features=n_b,
                    activation=activation,
                    batch_norm=False,
                    bias=False,
                )
            )
        # Final layer: no activation
        n_a, n_b = next(n_iter)
        ops.extend(
            make_layer(
                in_features=n_a,
                out_features=n_b,
                activation=None,
                batch_norm=False,
                bias=False,
            )
        )
        # Bundle into Sequential for forward pass
        self.ops = torch.nn.Sequential(*ops)

    def forward(self, x):
        return self.ops(x)


class clipped_ANN(torch.nn.Module):
    def __init__(self, N, range=[0, 2], activation=torch.nn.ReLU):
        """
        Packs sequence of artificial neural network layers in a list.
        """
        super().__init__()
        ops = []
        # Following recipe for itertools.pairwise
        NA, NB = itertools.tee(N)
        next(NB, None)
        n_iter = zip(NA, NB)
        # Create all but the last linear layer and activation
        for n_a, n_b in itertools.islice(n_iter, len(N) - 2):
            ops.extend(
                make_layer(
                    in_features=n_a,
                    out_features=n_b,
                    activation=activation,
                    batch_norm=False,
                    bias=False,
                )
            )
        # Final layer: no activation
        n_a, n_b = next(n_iter)
        ops.extend(
            make_layer(
                in_features=n_a,
                out_features=n_b,
                activation=None,
                batch_norm=False,
                bias=False,
            )
        )
        # Bundle into Sequential for forward pass
        self.ops = torch.nn.Sequential(*ops)

        self.min = range[0]
        self.max = range[1]

    def forward(self, x):
        return self.ops(x).clamp(min=self.min, max=self.max)
