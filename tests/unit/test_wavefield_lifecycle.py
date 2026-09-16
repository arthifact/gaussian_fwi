"""Owned disk scratch must survive neither a retained graph nor an ordinary failure."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from _problems import assert_exact, configuration, problem

import gaussian_fwi as fwi
from gaussian_fwi.core.wavefields import wavefield_directory


class WavefieldLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_failed_adjoint_cleans_real_storage_and_preserves_checkpoint_recovery(self):
        for accumulated in (False, True):
            # Footprints permit full-shot accumulation without altering the objective.
            from test_shot_accumulation import fixture

            field, data, partitions = fixture()
            config = configuration()
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                sentinel = root / "unrelated.txt"
                sentinel.write_bytes(b"preserve unrelated output")
                partial = fwi.invert(field, data, config, root / "prefix", partitions=partitions,
                                     max_updates=3, accumulate_shots=accumulated, wavefield_storage="disk")
                checkpoint = root / "prefix" / partial["checkpoint"]
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                held_graphs = []
                injected = RuntimeError("injected adjoint failure with a live disk graph")

                def fail_backward(tensor, *args, **kwargs):
                    files = list((root / "failed/wavefields").rglob("shot_*.bin"))
                    self.assertTrue(files)
                    self.assertTrue(all(p.stat().st_size > 0 for p in files))
                    held_graphs.append(tensor)
                    raise injected

                with patch.object(torch.Tensor, "backward", fail_backward):
                    with self.assertRaises(RuntimeError) as caught:
                        fwi.resume(checkpoint, data, root / "failed")
                self.assertIs(caught.exception, injected)
                self.assertTrue(held_graphs)
                self.assertEqual(list((root / "failed/wavefields").iterdir()), [])
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_hash)
                self.assertEqual(sentinel.read_bytes(), b"preserve unrelated output")
                self.assertFalse((root / "failed/report.json").exists())
                recovered, report = fwi.resume(checkpoint, data, root / "recovered")
                repeated, control = fwi.resume(checkpoint, data, root / "control")
                assert_exact(recovered.checkpoint(), repeated.checkpoint())
                assert_exact(report["solver_calls"], control["solver_calls"])
                assert_exact(report["topology_history"], control["topology_history"])
                self.assertEqual(list((root / "recovered/wavefields").iterdir()), [])

    def test_unrecognized_files_survive_and_cleanup_does_not_mask_original_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            injected = ValueError("original failure")
            with self.assertLogs("gaussian_fwi.core.wavefields", level="WARNING"):
                with self.assertRaises(ValueError) as caught:
                    with wavefield_directory(root, "disk") as scratch:
                        known = scratch / ("deepwave_tmp_123_" + "a" * 32)
                        known.mkdir()
                        (known / "shot_0.bin").write_bytes(b"owned scratch")
                        unrelated = scratch / "keep.txt"
                        unrelated.write_bytes(b"not a recognized scratch entry")
                        raise injected
            self.assertIs(caught.exception, injected)
            self.assertFalse(known.exists())
            self.assertEqual(unrelated.read_bytes(), b"not a recognized scratch entry")
            self.assertIn("Unrecognized wavefield entries preserved", injected.__notes__[0])
            with self.assertRaises(FileExistsError):
                with wavefield_directory(root, "disk"):
                    self.fail("Existing scratch directory must not be adopted")

    def test_replaced_directory_is_preserved_without_recursive_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                with wavefield_directory(root, "disk") as scratch:
                    moved = root / "original_scratch"
                    self.assertTrue(scratch.resolve().is_relative_to(root))
                    self.assertTrue(moved.resolve().is_relative_to(root))
                    scratch.rename(moved)
                    scratch.mkdir()
                    other = scratch / ("deepwave_tmp_123_" + "b" * 32)
                    other.mkdir()
                    sentinel = other / "shot_0.bin"
                    sentinel.write_bytes(b"different directory owner")
            self.assertEqual(sentinel.read_bytes(), b"different directory owner")
            self.assertTrue(moved.exists())

    def test_device_execution_does_not_create_disk_scratch(self):
        field, data, partitions = problem()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fit"
            fwi.invert(field, data, configuration(), output, partitions=partitions, max_updates=1)
            self.assertFalse((output / "wavefields").exists())
