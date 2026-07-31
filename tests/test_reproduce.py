from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reproduce


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def normalized_hash(data: bytes) -> str:
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def tiny_debd_manifest(contents: dict[str, bytes]) -> dict:
    splits = {}
    for split, data in contents.items():
        normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        lines = [line for line in normalized.split(b"\n") if line]
        splits[split] = {
            "file": f"toy.{split}.data",
            "rows": len(lines),
            "variables": len(lines[0].split(b",")),
            "sha256": hashlib.sha256(data).hexdigest(),
            "sha256_normalized_lf": normalized_hash(data),
        }
    return {
        "schema_version": 1,
        "dataset_count": 1,
        "datasets": {
            "toy": {
                "variables": 3,
                "splits": splits,
                "mle_pc_sha256": "not-used-in-import-test",
            }
        },
    }


class DebdImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo with spaces"
        self.root.mkdir()
        self.contents = {
            "train": b"0,1,0\n1,0,1\n",
            "valid": b"1,1,0\n",
            "test": b"0,0,1\n1,1,1\n",
        }
        write_json(
            self.root / "manifests" / "debd.json",
            tiny_debd_manifest(self.contents),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_source_directory(self, *, crlf: bool = False) -> Path:
        source = Path(self.temp.name) / "download" / "datasets" / "toy"
        source.mkdir(parents=True)
        for split, data in self.contents.items():
            if crlf:
                data = data.replace(b"\n", b"\r\n")
            (source / f"toy.{split}.data").write_bytes(data)
        return source.parents[1]

    def test_imports_extracted_directory_and_normalizes_hash_validation(self) -> None:
        source = self.make_source_directory(crlf=True)
        reproduce.import_debd(source, root=self.root)
        self.assertEqual([], reproduce.validate_debd_data(root=self.root))
        for split in self.contents:
            self.assertTrue(
                (
                    self.root
                    / "original_datasets"
                    / "toy"
                    / f"toy.{split}.data"
                ).is_file()
            )

    def test_imports_zip_archive(self) -> None:
        source = self.make_source_directory()
        archive_path = Path(self.temp.name) / "debd.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in source.rglob("*.data"):
                archive.write(path, Path("upstream") / "toy" / path.name)
        reproduce.import_debd(archive_path, root=self.root)
        self.assertEqual([], reproduce.validate_debd_data(root=self.root))

    def test_rejects_hash_mismatch(self) -> None:
        source = self.make_source_directory()
        (source / "datasets" / "toy" / "toy.test.data").write_bytes(b"1,0,0\n")
        with self.assertRaisesRegex(reproduce.ReproductionError, "hash mismatch"):
            reproduce.import_debd(source, root=self.root)
        self.assertFalse((self.root / "original_datasets").exists())

    def test_dry_run_does_not_write(self) -> None:
        source = self.make_source_directory()
        with contextlib.redirect_stdout(io.StringIO()):
            reproduce.import_debd(source, root=self.root, dry_run=True)
        self.assertFalse((self.root / "original_datasets").exists())


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.debd = reproduce.load_manifest("debd.json")
        cls.experiments = reproduce.load_manifest("experiments.json")
        cls.paper = reproduce.load_manifest("paper_results.json")

    def test_included_reference_inputs_are_complete_and_hashed(self) -> None:
        self.assertEqual(28, self.debd["dataset_count"])
        self.assertEqual(28, len(self.debd["datasets"]))
        for dataset, record in self.debd["datasets"].items():
            self.assertEqual({"train", "valid", "test"}, set(record["splits"]))
            pc = ROOT / "example_pcs" / f"{dataset}.json"
            self.assertEqual(
                record["mle_pc_sha256_normalized_lf"],
                reproduce.sha256_normalized_file(pc),
            )
        reference = ROOT / self.experiments["mnist"]["mle_pc"]["reference"]
        self.assertEqual(
            self.experiments["mnist"]["mle_pc"][
                "reference_sha256_normalized_lf"
            ],
            reproduce.sha256_normalized_file(reference),
        )

    def test_all_tpe_summaries_and_winners_are_present(self) -> None:
        summaries = list(
            (ROOT / "sweeps" / "tpe").glob("k*/*/study_summary.json")
        )
        self.assertEqual(85, len(summaries))
        for dataset in self.debd["datasets"]:
            for k in reproduce.EPSILONS:
                winner = reproduce.tpe_winner(dataset, k)
                self.assertGreater(winner["lr"], 0)
                self.assertGreater(winner["ratio"], 0)
                self.assertNotIn("\\", winner["summary"])
        self.assertEqual(1, reproduce.mnist_winner()["summary"].count("mnist"))

    def test_paper_dataset_orders_are_exact(self) -> None:
        self.assertEqual(
            [
                "accidents",
                "baudio",
                "c20ng",
                "connect4",
                "cr52",
                "jester",
                "kdd",
                "kosarek",
                "msnbc",
                "nltcs",
            ],
            self.paper["paper_tables"]["table1"]["dataset_order"],
        )
        self.assertEqual(
            [
                "baudio",
                "c20ng",
                "connect4",
                "cr52",
                "kdd",
                "kosarek",
                "msnbc",
                "nltcs",
            ],
            self.paper["paper_tables"]["table2"]["dataset_order"],
        )
        self.assertEqual(
            list(self.debd["datasets"]),
            self.paper["paper_tables"]["table5"]["dataset_order"],
        )

    def test_scientific_source_hashes_match_post_cleanup_snapshot(self) -> None:
        scientific = reproduce.load_manifest("scientific_code.json")
        authoritative = {
            "robustify.py",
            "peter.py",
            "peter_mnist.py",
            "learn_rltpm.py",
            "learn_mnist_pc.py",
            "attack.py",
            "attack_all.py",
            "random_corrupt.py",
            "fgsm.py",
            "eval.py",
            "eval_mnist.py",
            "learn_mle_adv.py",
            "prepare_mnist_data.py",
            "fgsm_mnist.py",
            "robustify_mnist.py",
            "tune.py",
            "tune_mnist.py",
            "sweep.py",
            "sweep_mnist.py",
            "best_sweep.py",
            "sweep_io.py",
        }
        self.assertEqual(authoritative, set(scientific["protected_files"]))
        post = scientific["post_cleanup_sha256_normalized_lf"]
        baseline = scientific["baseline_sha256_normalized_lf"]
        for name, expected in post.items():
            self.assertEqual(
                expected, reproduce.sha256_normalized_file(ROOT / name), name
            )
        changed = [
            name
            for name, before in baseline.items()
            if post[name] != before
        ]
        self.assertEqual(["learn_mle_adv.py"], changed)


class CommandConstructionTests(unittest.TestCase):
    def test_debd_peter_uses_included_winner_and_sys_executable(self) -> None:
        command = reproduce.build_debd_peter_commands(["nltcs"], [1])[0]
        winner = reproduce.tpe_winner("nltcs", 1)
        self.assertEqual(sys.executable, command.argv[0])
        self.assertEqual(str(ROOT / "peter.py"), command.argv[1])
        self.assertEqual("nltcs", command.argv[2])
        self.assertEqual(
            repr(float(winner["lr"])),
            command.argv[command.argv.index("--lr") + 1],
        )
        self.assertEqual(
            repr(float(winner["ratio"])),
            command.argv[command.argv.index("--ratio") + 1],
        )
        self.assertEqual(
            str(winner["iters"]),
            command.argv[command.argv.index("--iters") + 1],
        )
        self.assertEqual(
            "lr=0.111101_ratio=0.325155",
            command.skip_if.parent.name,
        )

    def test_forced_method_attacks_do_not_reuse_old_outputs(self) -> None:
        normal = reproduce.build_method_attack_commands(
            ["nltcs"], [1], force=False
        )
        forced = reproduce.build_method_attack_commands(
            ["nltcs"], [1], force=True
        )
        self.assertEqual(2, len(normal))
        self.assertTrue(all(command.skip_if is not None for command in normal))
        self.assertTrue(all(command.skip_if is None for command in forced))
        self.assertTrue(
            all(Path(command.argv[1]).name == "attack.py" for command in forced)
        )

    def test_argv_keeps_paths_with_spaces_as_single_arguments(self) -> None:
        fake_root = Path("C:/a repository/with spaces")
        argv = reproduce.py_command(
            "attack.py",
            fake_root / "a circuit.json",
            fake_root / "test data.csv",
            root=fake_root,
        )
        self.assertEqual(str(fake_root / "attack.py"), argv[1])
        self.assertEqual(str(fake_root / "a circuit.json"), argv[2])
        self.assertEqual(str(fake_root / "test data.csv"), argv[3])
        displayed = reproduce.display_command(argv)
        self.assertIn("a circuit.json", displayed)
        self.assertIn("test data.csv", displayed)

    def test_stage_order_is_explicit(self) -> None:
        self.assertEqual(
            (
                "import",
                "corrupt",
                "peter",
                "rltpm",
                "attack",
                "evaluate",
                "cw",
                "runtime",
            ),
            reproduce.DEBD_STAGE_ORDER,
        )
        self.assertEqual(
            ("prepare", "learn", "peter", "attack", "evaluate"),
            reproduce.MNIST_STAGE_ORDER,
        )

    def test_all_commands_execute_stages_in_declared_order(self) -> None:
        calls: list[str] = []
        debd_args = argparse.Namespace(
            source=Path("debd.zip"),
            force=False,
            dry_run=True,
            datasets=None,
            ks=None,
            jobs=1,
            skip_cw=False,
            skip_runtime=False,
        )
        with (
            patch.object(
                reproduce,
                "import_debd",
                side_effect=lambda *_args, **_kwargs: calls.append("import"),
            ),
            patch.object(
                reproduce,
                "run_debd_corrupt",
                side_effect=lambda _args: calls.append("corrupt"),
            ),
            patch.object(
                reproduce,
                "run_debd_peter",
                side_effect=lambda _args: calls.append("peter"),
            ),
            patch.object(
                reproduce,
                "run_debd_rltpm",
                side_effect=lambda _args: calls.append("rltpm"),
            ),
            patch.object(
                reproduce,
                "run_debd_attack",
                side_effect=lambda _args: calls.append("attack"),
            ),
            patch.object(
                reproduce,
                "run_debd_evaluate",
                side_effect=lambda _args: calls.append("evaluate"),
            ),
            patch.object(
                reproduce,
                "run_debd_cw",
                side_effect=lambda _args: calls.append("cw"),
            ),
            patch.object(
                reproduce,
                "run_debd_runtime",
                side_effect=lambda _args: calls.append("runtime"),
            ),
        ):
            reproduce.run_debd_all(debd_args)
        self.assertEqual(list(reproduce.DEBD_STAGE_ORDER), calls)

        calls.clear()
        mnist_args = argparse.Namespace(from_scratch=True)
        with (
            patch.object(
                reproduce,
                "run_mnist_prepare",
                side_effect=lambda _args: calls.append("prepare"),
            ),
            patch.object(
                reproduce,
                "run_mnist_learn",
                side_effect=lambda _args: calls.append("learn"),
            ),
            patch.object(
                reproduce,
                "run_mnist_peter",
                side_effect=lambda _args: calls.append("peter"),
            ),
            patch.object(
                reproduce,
                "run_mnist_attack",
                side_effect=lambda _args: calls.append("attack"),
            ),
            patch.object(
                reproduce,
                "run_mnist_evaluate",
                side_effect=lambda _args: calls.append("evaluate"),
            ),
        ):
            reproduce.run_mnist_all(mnist_args)
        self.assertEqual(list(reproduce.MNIST_STAGE_ORDER), calls)

    def test_existing_output_skips_without_launching_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            done = Path(temporary) / "done.json"
            done.write_text("{}", encoding="utf-8")
            command = reproduce.Command(
                "must skip",
                (sys.executable, "-c", "raise SystemExit(99)"),
                done,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = reproduce.run_command(command)
            self.assertEqual("skipped", result)

    def test_mnist_partial_run_restarts_and_atomic_marker_controls_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "sweeps" / "tpe" / "k1" / "mnist" / "study_summary.json"
            winner = {
                "lr": 1e-5,
                "ratio": 2.5,
                "final_adv_test_ll": -1.0,
                "trial_number": 4,
                "iters": 500,
                "n_trials_target": 50,
                "n_complete": 50,
                "n_failed": 0,
                "summary": "sweeps/tpe/k1/mnist/study_summary.json",
            }
            write_json(
                summary,
                {
                    "best": {
                        "lr": winner["lr"],
                        "ratio": winner["ratio"],
                    }
                },
            )
            write_json(
                root / "manifests" / "experiments.json",
                {
                    "mnist": {
                        "peter": {
                            "winner": winner,
                            "final_iterations": 1000,
                        }
                    }
                },
            )
            run_dir = (
                root
                / "results"
                / "mnist"
                / "k1"
                / reproduce.hyperparameter_dir(1e-5, 2.5)
            )
            run_dir.mkdir(parents=True)
            (run_dir / "circuit.json").write_text("{}", encoding="utf-8")
            write_json(run_dir / "config.json", {"iters": 500})
            restart = reproduce.build_mnist_peter_command(root=root)
            self.assertNotIn("--continue", restart.argv)
            self.assertIsNone(restart.skip_if)

            write_json(run_dir / "config.json", {"iters": 1000})
            write_json(run_dir / "metrics.json", {"final_lambda": 0.0})
            marker = reproduce.write_mnist_peter_marker(root=root)
            complete = reproduce.build_mnist_peter_command(root=root)
            self.assertNotIn("--continue", complete.argv)
            self.assertEqual(marker, complete.skip_if)

            forced = reproduce.build_mnist_peter_command(root=root, force=True)
            self.assertNotIn("--continue", forced.argv)
            self.assertIsNone(forced.skip_if)

            (run_dir / "circuit.json").write_text('{"changed": true}', encoding="utf-8")
            stale = reproduce.build_mnist_peter_command(root=root)
            self.assertIsNone(stale.skip_if)

    def test_canonical_artifact_names(self) -> None:
        commands = reproduce.artifact_commands(skip_runtime=False)
        arguments = {argument for command in commands for argument in command.argv}
        for name in (
            "table2.tex",
            "table4.tex",
            "table5.tex",
            "figure2.pdf",
            "figure3.pdf",
            "figure3_random.pdf",
            "figure3_adversarial.pdf",
            "figure4.png",
        ):
            self.assertTrue(
                any(Path(argument).name == name for argument in arguments),
                name,
            )

    def test_runtime_provenance_records_hardware_and_method(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            methodology = {
                "epsilon": 1,
                "warm_start_iterations": 100,
                "timed_max_iterations": 250,
                "timed_max_seconds": 30.0,
                "reported_iterations": 500,
            }
            write_json(
                root / "manifests" / "experiments.json",
                {"debd": {"runtime": methodology}},
            )
            with contextlib.redirect_stdout(io.StringIO()):
                output = reproduce.write_runtime_provenance(root=root)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(methodology, payload["methodology"])
            self.assertIn("os", payload["platform"])
            self.assertFalse(payload["numeric_equality_to_paper_required"])


class CliTests(unittest.TestCase):
    def run_quiet(self, argv: list[str]) -> int:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            return reproduce.main(argv)

    def test_help(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stdout(io.StringIO()):
                reproduce.build_parser().parse_args(["--help"])
        self.assertEqual(0, raised.exception.code)

    def test_every_stage_has_dependency_free_dry_run(self) -> None:
        commands = [
            ["--dry-run", "doctor"],
            [
                "--dry-run",
                "debd",
                "corrupt",
                "--dataset",
                "nltcs",
                "--k",
                "1",
            ],
            [
                "--dry-run",
                "debd",
                "peter",
                "--dataset",
                "nltcs",
                "--k",
                "1",
            ],
            [
                "--dry-run",
                "debd",
                "rltpm",
                "--dataset",
                "nltcs",
                "--k",
                "1",
            ],
            ["--dry-run", "debd", "attack"],
            [
                "--dry-run",
                "debd",
                "evaluate",
                "--dataset",
                "nltcs",
                "--k",
                "1",
            ],
            [
                "--dry-run",
                "debd",
                "cw",
                "--dataset",
                "nltcs",
                "--k",
                "1",
            ],
            ["--dry-run", "debd", "runtime"],
            [
                "--dry-run",
                "debd",
                "all",
                "--dataset",
                "nltcs",
                "--k",
                "1",
                "--skip-cw",
                "--skip-runtime",
            ],
            ["--dry-run", "mnist", "prepare"],
            ["--dry-run", "mnist", "learn"],
            ["--dry-run", "mnist", "peter"],
            ["--dry-run", "mnist", "attack"],
            ["--dry-run", "mnist", "evaluate"],
            ["--dry-run", "mnist", "all"],
            ["--dry-run", "artifacts", "--skip-runtime"],
            ["--dry-run", "verify-paper"],
        ]
        for argv in commands:
            with self.subTest(argv=argv):
                self.assertEqual(0, self.run_quiet(argv))


if __name__ == "__main__":
    unittest.main()
