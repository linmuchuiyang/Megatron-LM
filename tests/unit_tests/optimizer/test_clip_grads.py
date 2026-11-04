# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.optimizer import clip_grads
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == float('inf')


def _padded_grads(dtypes):
    # Keep a regressed fused kernel's wider dtype interpretation inside its allocation.
    # Padding also exposes writes beyond the logical gradient view.
    numel = 127  # Include a partial vector at the end of each gradient.
    bases = [torch.ones(4 * numel, device='cuda', dtype=dtype) for dtype in dtypes]
    grads = [base[:numel].fill_(1.0 / 64) for base in bases]
    return bases, grads


class TestMixedDtypeGradNormAndClip:
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    def teardown_method(self):
        Utils.destroy_model_parallel()

    # Fused kernels dispatch on the first tensor, so exercise both mixed-dtype orders.
    @pytest.mark.parametrize(
        'dtypes', [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)]
    )
    @pytest.mark.parametrize('layout', ['both', 'rank0_empty', 'disjoint', 'all_empty'])
    def test_grad_norm(self, dtypes, layout):
        _, grads = _padded_grads(dtypes)
        rank = torch.distributed.get_rank()
        if layout == 'all_empty' or (layout == 'rank0_empty' and rank == 0):
            grads = []
        elif layout == 'disjoint':
            grads = [grads[rank % 2]]
        squared_norm = torch.zeros(1, device='cuda', dtype=torch.float64)
        for grad in grads:
            squared_norm += grad.double().square().sum()
        torch.distributed.all_reduce(squared_norm)

        actual = clip_grads.get_grad_norm_fp32(
            grads, grad_stats_parallel_group=torch.distributed.group.WORLD
        )

        torch.testing.assert_close(float(actual), float(squared_norm.sqrt()), rtol=2e-6, atol=1e-8)

    # Scaling also dispatches on the first tensor; both mixed orders must preserve padding.
    @pytest.mark.parametrize(
        'dtypes', [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)]
    )
    @pytest.mark.parametrize('tensor_coefficient', [False, True])
    def test_clip_gradients(self, dtypes, tensor_coefficient):
        bases, grads = _padded_grads(dtypes)
        params = []
        for grad in grads:
            # The dtype of a decoupled gradient need not match its parameter's dtype.
            param = torch.nn.Parameter(torch.zeros_like(grad, dtype=torch.bfloat16))
            param.decoupled_grad = grad
            params.append(param)
        total_norm = torch.tensor([2.0], device='cuda') if tensor_coefficient else 2.0
        expected = [base.clone() for base in bases]
        for base, grad in zip(expected, grads):
            base[: grad.numel()].mul_(0.5)

        # Supply the norm independently to catch scaling failures even with a correct norm.
        clip_grads.clip_grad_by_total_norm_fp32(
            params, 1.0000005, total_norm, use_decoupled_grad=True
        )

        for actual, reference in zip(bases, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    @pytest.mark.parametrize('tensor_coefficient', [False, True])
    def test_clip_empty_gradients(self, tensor_coefficient):
        total_norm = torch.tensor([2.0], device='cuda') if tensor_coefficient else 2.0
        clip_grads.clip_grad_by_total_norm_fp32([], 1.0, total_norm, use_decoupled_grad=True)
        torch.cuda.synchronize()
