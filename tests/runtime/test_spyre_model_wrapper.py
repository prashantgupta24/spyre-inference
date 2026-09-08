# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for _SpyreModelWrapper input/output conversion."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spyre_inference.custom_ops import register_all
from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper


class _Capture(nn.Module):
    """Records the tensors it receives and returns a dummy tensor."""

    def __init__(self):
        super().__init__()
        self.captured_args: tuple | None = None
        self.captured_kwargs: dict | None = None

    def forward(self, *args, **kwargs):
        self.captured_args = args
        self.captured_kwargs = kwargs
        return torch.zeros(1)


@pytest.fixture(scope="module", autouse=True)
def _register_ops():
    register_all()


def _assert_int_on_spyre(t: torch.Tensor) -> None:
    assert isinstance(t, torch.Tensor)
    assert t.device.type == "spyre"
    assert t.dtype == torch.int64


def test_nested_int_tensors_in_args_are_moved_to_spyre():
    spyre_device = torch.device("spyre")
    capture = _Capture()
    wrapper = _SpyreModelWrapper(capture, spyre_device, keep_outputs_on_device=True)

    nested = [torch.tensor([1, 2], dtype=torch.int32, device="cpu")]
    wrapper(nested)

    assert capture.captured_args is not None
    out_nested = capture.captured_args[0]
    assert isinstance(out_nested, list)
    _assert_int_on_spyre(out_nested[0])


def test_nested_int_tensors_in_kwargs_are_moved_to_spyre():
    spyre_device = torch.device("spyre")
    capture = _Capture()
    wrapper = _SpyreModelWrapper(capture, spyre_device, keep_outputs_on_device=True)

    kwargs = {
        "encoder_outputs": {
            "ids": torch.tensor([3, 4], dtype=torch.int64, device="cpu"),
        },
    }
    wrapper(**kwargs)

    assert capture.captured_kwargs is not None
    ids = capture.captured_kwargs["encoder_outputs"]["ids"]
    _assert_int_on_spyre(ids)


def test_non_int_tensors_are_not_relocated():
    spyre_device = torch.device("spyre")
    capture = _Capture()
    wrapper = _SpyreModelWrapper(capture, spyre_device, keep_outputs_on_device=True)

    float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cpu")
    int_tensor = torch.tensor([1, 2], dtype=torch.int32, device="cpu")
    wrapper(float_tensor, int_tensor)

    assert capture.captured_args is not None
    out_float, out_int = capture.captured_args
    assert out_float is float_tensor
    assert out_float.device.type == "cpu"
    _assert_int_on_spyre(out_int)


def test_outputs_are_moved_to_cpu_by_default():
    spyre_device = torch.device("spyre")
    capture = _Capture()
    wrapper = _SpyreModelWrapper(capture, spyre_device, keep_outputs_on_device=False)

    output = wrapper(torch.tensor([1, 2], dtype=torch.int32, device="cpu"))

    assert output.device.type == "cpu"
