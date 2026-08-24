"""Device selection, and the float64 policy that MPS forces.

The interesting tests here are not "does it pick MPS on a Mac". They are the
ones that pin down behaviour nobody would notice going wrong: that an explicit
request for an unavailable device raises instead of falling back, that the
float policy is recorded rather than assumed, and that `pin_to_cpu` refuses an
explicit request while substituting an automatic one.

Availability is faked throughout. A test whose result depends on what hardware
the developer happens to have is not a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from vitruve.models import device as dev


@pytest.fixture
def fake_availability(monkeypatch):
    """Control which devices `resolve` believes in."""

    def install(**present: bool) -> None:
        def always(value: bool):
            return lambda: value

        table = {k: always(present.get(k, False)) for k in dev.VALID}
        monkeypatch.setattr(dev, "AVAILABILITY", table)

    return install


def test_preference_order_is_mps_then_cuda_then_cpu(fake_availability):
    fake_availability(mps=True, cuda=True, cpu=True)
    assert dev.resolve({}.get("x")).kind == "mps"

    fake_availability(mps=False, cuda=True, cpu=True)
    assert dev.resolve(env={}).kind == "cuda"

    fake_availability(mps=False, cuda=False, cpu=True)
    assert dev.resolve(env={}).kind == "cpu"


def test_automatic_selection_records_what_it_skipped(fake_availability):
    fake_availability(mps=False, cuda=False, cpu=True)
    resolved = dev.resolve(env={})
    assert not resolved.explicit
    # The manifest has to be able to say why a slow device was used, or a
    # report that took ten times as long looks like a regression in the code.
    assert "skipped mps, cuda" in resolved.description


def test_env_override_is_honoured(fake_availability):
    fake_availability(mps=True, cuda=False, cpu=True)
    resolved = dev.resolve(env={dev.DEVICE_ENV_VAR: "cpu"})
    assert resolved.kind == "cpu"
    assert resolved.explicit


def test_explicit_request_beats_the_environment(fake_availability):
    fake_availability(mps=True, cuda=False, cpu=True)
    assert dev.resolve("mps", env={dev.DEVICE_ENV_VAR: "cpu"}).kind == "mps"


def test_an_unavailable_explicit_request_raises_rather_than_falling_back(fake_availability):
    """The whole point of the override is reproducing a specific run.

    A silent fallback would produce numbers from a different device under the
    same manifest entry, which is worse than failing.
    """
    fake_availability(mps=False, cuda=False, cpu=True)
    with pytest.raises(dev.DeviceUnavailable, match="not available"):
        dev.resolve("mps", env={})
    with pytest.raises(dev.DeviceUnavailable, match="not available"):
        dev.resolve(env={dev.DEVICE_ENV_VAR: "cuda"})


def test_an_unknown_device_name_raises(fake_availability):
    fake_availability(cpu=True)
    with pytest.raises(dev.DeviceUnavailable, match="not a device"):
        dev.resolve("tpu", env={})


def test_mps_declares_that_it_cannot_carry_float64(fake_availability):
    """Metal has no 64-bit float type, and the geometry layer is float64.

    This flag is what stops somebody moving a coordinate array onto the device
    to save a copy and quantising every confidence interval in the report.
    """
    fake_availability(mps=True, cpu=True)
    resolved = dev.resolve(env={})
    assert resolved.kind == "mps"
    assert resolved.supports_float64 is False
    assert "float64" in resolved.description

    fake_availability(mps=False, cuda=False, cpu=True)
    assert dev.resolve(env={}).supports_float64 is True


def test_the_float_policy_is_the_same_sentence_everywhere(fake_availability):
    fake_availability(cpu=True)
    resolved = dev.resolve(env={})
    env = dev.describe_environment(resolved)
    assert env["float_policy"] == "models float32, geometry float64 on host"
    assert env["device"] == "cpu"
    assert set(env) >= {"device", "device_description", "platform", "python", "numpy", "torch"}


def test_to_host_widens_to_float64():
    """Every backend result crosses this function on its way to the geometry."""
    arr = np.array([[1.5, 2.5]], dtype=np.float32)
    out = dev.to_host(arr)
    assert out.dtype == np.float64

    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[1.5, 2.5]], dtype=torch.float32)
    out = dev.to_host(tensor)
    assert out.dtype == np.float64
    assert out.shape == (1, 2)


def test_to_host_detaches_a_grad_tracking_tensor():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.0, 2.0], requires_grad=True) * 2
    assert tensor.requires_grad
    out = dev.to_host(tensor)
    assert out.tolist() == [2.0, 4.0]


def test_pin_to_cpu_substitutes_automatically_and_says_so(fake_availability):
    """Two backends are CPU-only because upstream mistook MPS for CUDA."""
    fake_availability(mps=True, cpu=True)
    resolved, note = dev.pin_to_cpu(None, "because upstream says so")
    assert resolved.kind == "cpu"
    assert "mps was available but not used" in note
    assert "because upstream says so" in note


def test_pin_to_cpu_is_silent_when_cpu_was_the_answer_anyway(fake_availability):
    fake_availability(mps=False, cuda=False, cpu=True)
    resolved, note = dev.pin_to_cpu(None, "reason")
    assert resolved.kind == "cpu"
    assert note == ""


def test_pin_to_cpu_refuses_an_explicit_accelerator(fake_availability):
    """An explicit request that is silently downgraded is a lie about timings."""
    fake_availability(mps=True, cpu=True)
    mps = dev.resolve("mps", env={})
    assert mps.explicit
    with pytest.raises(dev.DeviceUnavailable, match="Pass the CPU device instead"):
        dev.pin_to_cpu(mps, "reason")


def test_pin_to_cpu_substitutes_an_already_resolved_automatic_device(fake_availability):
    """The pipeline resolves once and threads the object through every backend.

    By then there is no `None` left to mean "nobody asked", so the substitution
    has to key off `explicit` or a pinned backend refuses every default run.
    """
    fake_availability(mps=True, cpu=True)
    auto = dev.resolve(env={})
    assert auto.kind == "mps" and not auto.explicit
    resolved, note = dev.pin_to_cpu(auto, "reason")
    assert resolved.kind == "cpu"
    assert "mps was available but not used" in note


def test_pin_to_cpu_passes_an_explicit_cpu_through_untouched(fake_availability):
    fake_availability(mps=True, cpu=True)
    cpu = dev.resolve("cpu", env={})
    resolved, note = dev.pin_to_cpu(cpu, "reason")
    assert resolved is cpu
    assert note == ""


def test_device_rejects_a_name_outside_the_vocabulary():
    with pytest.raises(ValueError, match="unknown device"):
        dev.Device(kind="gpu", description="", explicit=False, supports_float64=True)
