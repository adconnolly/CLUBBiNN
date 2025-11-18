"""Unit tests for ANN and Clipped_ANN architectures."""

import numpy as np
import pytest
import torch

from subgrid_parameterization.arch import ANN, Clipped_ANN


class TestANN:
    """Tests for ANN and Clipped_ANN classes."""

    def test_ann_forward(self):
        """Test ANN forward pass returns correct shape and tensor type."""
        N = [4, 16, 8, 2]
        model = ANN(N)
        x = torch.randn(10, N[0])
        out = model(x)

        assert out.shape == (10, N[-1])
        assert isinstance(out, torch.Tensor)

    def test_clipped_ann_default_range(self):
        """Test Clipped_ANN uses default clamp range if none is provided."""
        N = [4, 8, 2]
        model = Clipped_ANN(N)
        x = torch.randn(6, N[0]) * 10
        out = model(x)
        # Default clamp range is [0, 2]
        torch.testing.assert_close(out, out.clamp(0, 2))

    def test_clipped_ann_forward(self):
        """
        Test Clipped_ANN forward pass clamps output within specified range.

        Could be improved to verify that the clamping is active in the net
        rather that assuming.
        """
        N = [4, 16, 8, 2]
        clamp_range = [0.0, 1.0]
        model = Clipped_ANN(N, clamping_range=clamp_range)
        x = torch.randn(10, N[0]) * 10
        out = model(x)

        assert out.shape == (10, N[-1])
        torch.testing.assert_close(out, out.clamp(*clamp_range))

    def test_ann_linear_last_layer(self):
        """Test ANN last layer is Linear (no activation applied)."""
        N = [4, 16, 8, 2]
        model = ANN(N)
        last_layer = list(model.ops.children())[-1]

        assert isinstance(last_layer, torch.nn.Linear)

    def test_ann_edge_cases(self):
        """Test ANN with very short and long layer lists."""
        # Short: input and output only
        N_short = [4, 2]
        model_short = ANN(N_short)
        x_short = torch.randn(5, N_short[0])
        out_short = model_short(x_short)
        torch.testing.assert_close(
            torch.tensor(out_short.shape), torch.tensor([5, N_short[-1]])
        )

        # Long: many layers
        N_long = [4] + [8] * 10 + [2]
        model_long = ANN(N_long)
        x_long = torch.randn(3, N_long[0])
        out_long = model_long(x_long)
        torch.testing.assert_close(
            torch.tensor(out_long.shape), torch.tensor([3, N_long[-1]])
        )
