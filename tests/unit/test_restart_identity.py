"""Actual-content restart identity, independent of caller-supplied labels."""

import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch
from _problems import assert_exact, configuration, problem

import gaussian_fwi
from gaussian_fwi.core import GridSpec, Observations
from gaussian_fwi.core.footprints import FootprintAcquisition, GaussianFootprint
from gaussian_fwi.core.identity import observation_sha256


class RestartIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, module, *, footprint=False, label=""):
        field, data, partitions = problem()
        field.add_grid_level((2, 2))
        if footprint:
            acquisition = FootprintAcquisition(data.acquisition, GaussianFootprint(5., 0.))
            with torch.no_grad():
                data = Observations(acquisition, acquisition.simulate(field()+45))
        data.sha256 = label
        config = replace(configuration(), seed_shape=None, steps_per_stage=2, validation_interval=1)
        return field, data, partitions, config

    def reject(self, module, checkpoint, data, destination, message="observation.*not match"):
        before = data.acquisition.counts
        with self.assertRaisesRegex(ValueError, message):
            module.resume(checkpoint, data, destination)
        self.assertEqual(data.acquisition.counts, before)
        self.assertFalse(destination.exists())

    def test_changed_content_is_rejected_with_empty_and_stale_nonempty_labels(self):
        for module in (gaussian_fwi,):
            for label in ("", "stale-caller-label"):
                with self.subTest(api=module.__name__, label=label), tempfile.TemporaryDirectory() as tmp:
                    field, data, split, config = self.fixture(module, label=label)
                    root = Path(tmp)
                    module.invert(field, data, config, root / "fit", partitions=split)
                    variants = []
                    for partition in split:
                        changed = deepcopy(data)
                        changed.traces[:, split[partition]] *= 1.5
                        variants.append((f"{partition}_traces", changed))
                    for name in ("source_amplitudes", "source_locations", "receiver_locations"):
                        changed = deepcopy(data)
                        getattr(changed.acquisition, name).flatten()[0].add_(1)
                        variants.append((name, changed))
                    for name, value in (("dt", .0009), ("accuracy", 8), ("pml_width", 7),
                                        ("pml_frequency", 24.), ("max_velocity", 4600.),
                                        ("grid", GridSpec((12, 12), 11.))):
                        changed = deepcopy(data)
                        setattr(changed.acquisition, name, value)
                        variants.append((name, changed))
                    for name, changed in variants:
                        with self.subTest(change=name):
                            self.assertEqual(changed.sha256, data.sha256)
                            self.reject(module, root / "fit/stage_00.pt", changed, root / name)

    def test_matching_clones_resume_exactly_for_point_and_finite_measurements(self):
        for module in (gaussian_fwi,):
            for finite in (False, True):
                with self.subTest(api=module.__name__, finite=finite), tempfile.TemporaryDirectory() as tmp:
                    field, data, split, config = self.fixture(module, footprint=finite)
                    root = Path(tmp)
                    original = module.invert(field, data, config, root / "fit", partitions=split)
                    acquisition = (FootprintAcquisition.from_checkpoint(data.acquisition.checkpoint()) if finite
                                   else replace(data.acquisition, source_amplitudes=data.acquisition.source_amplitudes.clone()))
                    # Different layout, object identity, counters and descriptive metadata.
                    traces = data.traces.transpose(1, 2).contiguous().transpose(1, 2)
                    clone = Observations(acquisition, traces, metadata={"description": "clone"})
                    restored, report = module.resume(root / "fit/stage_00.pt", clone, root / "resume")
                    assert_exact(restored.checkpoint(), field.checkpoint())
                    for key in ("solver_calls", "waveforms", "stages", "observation_identity"):
                        assert_exact(report[key], original[key])
                    a = torch.load(root / "fit/stage_01.pt", weights_only=True)
                    b = torch.load(root / "resume/stage_01.pt", weights_only=True)
                    for key in ("field", "optimizer", "stages", "solver_calls", "observation_identity",
                                "topology_state", "topology_history"):
                        assert_exact(a.get(key), b.get(key))
                    # Wall time is deliberately different; numerical history is exact.
                    def history(payload):
                        return [{k: v for k, v in row.items() if k != "elapsed_s"} for row in payload["history"]]
                    assert_exact(history(a), history(b))
                    prior = torch.load(root / "fit/stage_00.pt", weights_only=True)["history"]
                    assert_exact(b["history"][:len(prior)], prior)
                    if finite:
                        changed = Observations(FootprintAcquisition(data.acquisition.base, GaussianFootprint(4., 0.)), traces)
                        self.reject(module, root / "fit/stage_00.pt", changed, root / "changed_footprint")

    def test_legacy_and_unknown_identities_are_rejected_even_with_matching_labels(self):
        for module in (gaussian_fwi,):
            with self.subTest(api=module.__name__), tempfile.TemporaryDirectory() as tmp:
                field, data, split, config = self.fixture(module)
                data.sha256 = observation_sha256(data.acquisition, data.traces)
                root = Path(tmp)
                module.invert(field, data, config, root / "fit", partitions=split)
                original = torch.load(root / "fit/stage_00.pt", weights_only=True)
                for name in ("missing", "unknown", "changed_digest"):
                    payload = deepcopy(original)
                    if name == "missing":
                        payload.pop("observation_identity", None)
                        message = "observation.*not match"
                    else:
                        payload["observation_identity"] = {"format": "unknown" if name == "unknown" else "gaussian-fwi-observation-content-v1",
                                                           "sha256": "0"*64}
                        message = "observation.*not match"
                    checkpoint = root / f"{name}.pt"
                    torch.save(payload, checkpoint)
                    self.reject(module, checkpoint, data, root / name, message)
