"""Reviewer-facing orchestration for the paper experiments.

The numerical implementations live in the original scripts.  This CLI only
validates inputs, reads the checked-in manifests, and launches those scripts
with ``sys.executable`` from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "manifests"
OUTPUT_DIR = ROOT / "paper_outputs"
EPSILONS = (1, 3, 5)

DEBD_STAGE_ORDER = (
    "import",
    "corrupt",
    "peter",
    "rltpm",
    "attack",
    "evaluate",
    "cw",
    "runtime",
)
MNIST_STAGE_ORDER = ("prepare", "learn", "peter", "attack", "evaluate")


class ReproductionError(RuntimeError):
    """A concise, reviewer-actionable workflow error."""


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    skip_if: Path | None = None


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReproductionError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReproductionError(f"Invalid JSON in {path}: {exc}") from exc


def load_manifest(name: str, *, root: Path = ROOT) -> dict[str, Any]:
    return load_json(root / "manifests" / name)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def py_command(script: str, *args: object, root: Path = ROOT) -> tuple[str, ...]:
    return (sys.executable, str(root / script), *(str(arg) for arg in args))


def display_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(argv))
    return shlex.join(argv)


def display_path(path: Path, *, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def run_command(command: Command, *, dry_run: bool = False) -> str:
    if command.skip_if is not None and command.skip_if.is_file():
        print(f"skip  {command.label}: {display_path(command.skip_if)} exists")
        return "skipped"
    print(f"run   {command.label}")
    print(f"  $ {display_command(command.argv)}", flush=True)
    if dry_run:
        return "dry-run"
    env = os.environ.copy()
    # Keep Windows consoles from failing on Greek epsilons in producer logs.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(command.argv, cwd=ROOT, check=False, env=env)
    if completed.returncode != 0:
        raise ReproductionError(
            f"{command.label} failed with exit code {completed.returncode}"
        )
    return "completed"


def run_commands(
    commands: Sequence[Command], *, jobs: int = 1, dry_run: bool = False
) -> list[str]:
    if jobs < 1:
        raise ReproductionError("--jobs must be at least 1")
    if dry_run or jobs == 1 or len(commands) < 2:
        return [run_command(command, dry_run=dry_run) for command in commands]

    results: list[str] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_command, command, dry_run=False): command
            for command in commands
        }
        for future in as_completed(futures):
            command = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # subprocess failures are reported together
                failures.append(f"{command.label}: {exc}")
    if failures:
        raise ReproductionError(
            f"{len(failures)} job(s) failed:\n  " + "\n  ".join(failures)
        )
    return results


def normalized_text_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_file(path: Path) -> str:
    return sha256_bytes(normalized_text_bytes(path.read_bytes()))


def debd_dataset_names(*, root: Path = ROOT) -> list[str]:
    manifest = load_manifest("debd.json", root=root)
    return list(manifest["datasets"])


def select_datasets(
    requested: Sequence[str] | None, *, root: Path = ROOT
) -> list[str]:
    available = debd_dataset_names(root=root)
    if not requested:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ReproductionError(
            f"Unknown DEBD dataset(s): {', '.join(unknown)}. "
            f"Expected one of: {', '.join(available)}"
        )
    return list(dict.fromkeys(requested))


def select_ks(requested: Sequence[int] | None) -> list[int]:
    return list(dict.fromkeys(requested)) if requested else list(EPSILONS)


def validate_debd_data(
    *,
    root: Path = ROOT,
    manifest: dict[str, Any] | None = None,
    datasets: Sequence[str] | None = None,
) -> list[str]:
    manifest = manifest or load_manifest("debd.json", root=root)
    selected = datasets or list(manifest["datasets"])
    errors: list[str] = []
    for dataset in selected:
        record = manifest["datasets"][dataset]
        expected_variables = int(record["variables"])
        for split, split_record in record["splits"].items():
            path = root / "original_datasets" / dataset / split_record["file"]
            if not path.is_file():
                errors.append(f"missing {path.relative_to(root)}")
                continue
            raw = path.read_bytes()
            normalized = normalized_text_bytes(raw)
            actual_hash = sha256_bytes(normalized)
            expected_hash = split_record["sha256_normalized_lf"]
            if actual_hash != expected_hash:
                errors.append(
                    f"hash mismatch {path.relative_to(root)}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
                continue
            lines = [line for line in normalized.split(b"\n") if line]
            variables = len(lines[0].split(b",")) if lines else 0
            if len(lines) != int(split_record["rows"]):
                errors.append(
                    f"row-count mismatch {path.relative_to(root)}: "
                    f"expected {split_record['rows']}, got {len(lines)}"
                )
            if variables != expected_variables:
                errors.append(
                    f"width mismatch {path.relative_to(root)}: "
                    f"expected {expected_variables}, got {variables}"
                )
    return errors


def require_debd_inputs(datasets: Sequence[str], *, root: Path = ROOT) -> None:
    errors = validate_debd_data(root=root, datasets=datasets)
    for dataset in datasets:
        pc = root / "example_pcs" / f"{dataset}.json"
        if not pc.is_file():
            errors.append(f"missing {pc.relative_to(root)}")
    if errors:
        raise ReproductionError(
            "DEBD preflight failed:\n  "
            + "\n  ".join(errors[:20])
            + ("\n  ..." if len(errors) > 20 else "")
            + "\nRun `python reproduce.py debd import --source PATH` first."
        )


def _expected_source_files(
    manifest: dict[str, Any],
) -> dict[str, tuple[str, str, dict[str, Any]]]:
    expected: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for dataset, dataset_record in manifest["datasets"].items():
        for split, split_record in dataset_record["splits"].items():
            expected[split_record["file"]] = (dataset, split, split_record)
    return expected


def _directory_source_bytes(
    source: Path, expected_names: set[str]
) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in source.rglob("*"):
        if not path.is_file() or path.name not in expected_names:
            continue
        data = path.read_bytes()
        if path.name in found and found[path.name] != data:
            raise ReproductionError(
                f"Archive source has conflicting copies of {path.name}"
            )
        found[path.name] = data
    return found


def _zip_source_bytes(source: Path, expected_names: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if member.is_dir() or name not in expected_names:
                continue
            data = archive.read(member)
            if name in found and found[name] != data:
                raise ReproductionError(
                    f"Archive source has conflicting copies of {name}"
                )
            found[name] = data
    return found


def _tar_source_bytes(source: Path, expected_names: set[str]) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with tarfile.open(source, mode="r:*") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if not member.isfile() or name not in expected_names:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            data = stream.read()
            if name in found and found[name] != data:
                raise ReproductionError(
                    f"Archive source has conflicting copies of {name}"
                )
            found[name] = data
    return found


def read_debd_source(source: Path, expected_names: set[str]) -> dict[str, bytes]:
    source = source.expanduser().resolve()
    if source.is_dir():
        return _directory_source_bytes(source, expected_names)
    if not source.is_file():
        raise ReproductionError(f"DEBD source does not exist: {source}")
    if zipfile.is_zipfile(source):
        return _zip_source_bytes(source, expected_names)
    if tarfile.is_tarfile(source):
        return _tar_source_bytes(source, expected_names)
    raise ReproductionError(
        f"Unsupported DEBD source {source}; use an extracted directory, zip, or tar archive."
    )


def import_debd(
    source: Path,
    *,
    root: Path = ROOT,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    manifest = load_manifest("debd.json", root=root)
    expected = _expected_source_files(manifest)
    found = read_debd_source(source, set(expected))
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ReproductionError(
            f"DEBD source is missing {len(missing)} required split file(s): "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    validated: dict[str, bytes] = {}
    for filename, (_dataset, _split, record) in expected.items():
        data = found[filename]
        actual = sha256_bytes(normalized_text_bytes(data))
        expected_hash = record["sha256_normalized_lf"]
        if actual != expected_hash:
            raise ReproductionError(
                f"DEBD source hash mismatch for {filename}: "
                f"expected {expected_hash}, got {actual}"
            )
        validated[filename] = data

    writes: list[tuple[Path, bytes]] = []
    skips: list[Path] = []
    for filename, (dataset, _split, record) in expected.items():
        destination = root / "original_datasets" / dataset / filename
        if destination.is_file() and not force:
            existing = sha256_bytes(normalized_text_bytes(destination.read_bytes()))
            if existing == record["sha256_normalized_lf"]:
                skips.append(destination)
                continue
            raise ReproductionError(
                f"{destination.relative_to(root)} exists but is invalid; rerun with --force."
            )
        writes.append((destination, validated[filename]))

    for destination in skips:
        print(f"skip  {destination.relative_to(root)} (already valid)")
    for destination, data in writes:
        print(f"write {destination.relative_to(root)}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(destination)

    if dry_run:
        print("DEBD import dry-run complete; no files written.")
        return
    errors = validate_debd_data(root=root, manifest=manifest)
    if errors:
        raise ReproductionError("Imported DEBD validation failed:\n  " + "\n  ".join(errors))
    print(f"Validated all {manifest['dataset_count']} DEBD datasets.")


def tpe_winner(dataset: str, k: int, *, root: Path = ROOT) -> dict[str, Any]:
    experiments = load_manifest("experiments.json", root=root)
    try:
        winner = (
            experiments["mnist"]["peter"]["winner"]
            if dataset == "mnist"
            else experiments["debd"]["peter"]["winners"][dataset][str(k)]
        )
    except KeyError as exc:
        raise ReproductionError(
            f"No included TPE winner for {dataset}, epsilon={k}"
        ) from exc
    summary = root / winner["summary"]
    if not summary.is_file():
        raise ReproductionError(f"Missing included TPE summary: {summary}")
    live = load_json(summary).get("best")
    if not isinstance(live, dict):
        raise ReproductionError(f"TPE summary has no successful best trial: {summary}")
    for field in ("lr", "ratio"):
        if float(live[field]) != float(winner[field]):
            raise ReproductionError(
                f"Manifest/TPE mismatch for {dataset} epsilon={k} field {field}"
            )
    return winner


def hyperparameter_dir(lr: float, ratio: float) -> str:
    return f"lr={lr:g}_ratio={ratio:g}"


def debd_peter_output(
    dataset: str, k: int, winner: dict[str, Any], *, root: Path = ROOT
) -> Path:
    return (
        root
        / "results"
        / dataset
        / f"k{k}"
        / hyperparameter_dir(float(winner["lr"]), float(winner["ratio"]))
    )


def build_random_corruption_commands(
    datasets: Sequence[str],
    ks: Sequence[int],
    *,
    jobs: int,
    force: bool,
    root: Path = ROOT,
) -> list[Command]:
    commands: list[Command] = []
    for k in ks:
        argv = list(py_command("random_corrupt.py", "--k", k, "-j", jobs, root=root))
        for dataset in datasets:
            argv.extend(("--dataset", dataset))
        if force:
            argv.append("--force")
        commands.append(Command(f"random corruptions epsilon={k}", tuple(argv)))
    return commands


def build_mle_attack_commands(
    datasets: Sequence[str],
    ks: Sequence[int],
    *,
    force: bool,
    root: Path = ROOT,
) -> list[Command]:
    commands: list[Command] = []
    for k in ks:
        for dataset in datasets:
            output = root / "adversarial_datasets" / f"{dataset}_K{k}.data"
            command = Command(
                f"MLE attack {dataset} epsilon={k}",
                py_command(
                    "attack.py",
                    root / "example_pcs" / f"{dataset}.json",
                    root
                    / "original_datasets"
                    / dataset
                    / f"{dataset}.test.data",
                    output,
                    k,
                    root=root,
                ),
                None if force else output,
            )
            commands.append(command)
    return commands


def build_debd_peter_commands(
    datasets: Sequence[str],
    ks: Sequence[int],
    *,
    force: bool = False,
    root: Path = ROOT,
) -> list[Command]:
    commands: list[Command] = []
    for k in ks:
        for dataset in datasets:
            winner = tpe_winner(dataset, k, root=root)
            output = debd_peter_output(dataset, k, winner, root=root)
            command = Command(
                f"PeTeR {dataset} epsilon={k}",
                py_command(
                    "peter.py",
                    dataset,
                    "--k",
                    k,
                    "--lr",
                    repr(float(winner["lr"])),
                    "--ratio",
                    repr(float(winner["ratio"])),
                    "--iters",
                    int(winner["iters"]),
                    root=root,
                ),
                None if force else output / "circuit.json",
            )
            commands.append(command)
    return commands


def build_rltpm_commands(
    datasets: Sequence[str],
    ks: Sequence[int],
    *,
    jobs: int,
    root: Path = ROOT,
) -> list[Command]:
    commands: list[Command] = []
    for k in ks:
        argv = list(
            py_command("learn_rltpm.py", "--k", k, "-j", jobs, root=root)
        )
        if datasets:
            argv.append("--datasets")
            argv.extend(datasets)
        commands.append(Command(f"RL-TPM epsilon={k}", tuple(argv)))
    return commands


def rltpm_circuit_path(dataset: str, k: int, *, root: Path = ROOT) -> Path:
    return (
        root
        / "rltpm_learned_pcs"
        / "hclt"
        / dataset
        / "4"
        / f"K{k}"
        / f"hclt_{dataset}_blocksize4_seed0.json"
    )


def build_method_attack_commands(
    datasets: Sequence[str],
    ks: Sequence[int],
    *,
    force: bool,
    root: Path = ROOT,
) -> list[Command]:
    commands: list[Command] = []
    for k in ks:
        for dataset in datasets:
            original = (
                root
                / "original_datasets"
                / dataset
                / f"{dataset}.test.data"
            )
            winner = tpe_winner(dataset, k, root=root)
            peter_dir = debd_peter_output(dataset, k, winner, root=root)
            rltpm_dir = rltpm_circuit_path(dataset, k, root=root).parent
            for method, circuit, output in (
                (
                    "PeTeR",
                    peter_dir / "circuit.json",
                    peter_dir / f"{dataset}_K{k}_peter.data",
                ),
                (
                    "RL-TPM",
                    rltpm_circuit_path(dataset, k, root=root),
                    rltpm_dir / f"{dataset}_K{k}_rltpm.data",
                ),
            ):
                commands.append(
                    Command(
                        f"{method} attack {dataset} epsilon={k}",
                        py_command(
                            "attack.py",
                            circuit,
                            original,
                            output,
                            k,
                            root=root,
                        ),
                        None if force else output,
                    )
                )
    return commands


def build_eval_commands(
    datasets: Sequence[str],
    ks: Sequence[int],
    *,
    jobs: int,
    force: bool,
    root: Path = ROOT,
) -> list[Command]:
    commands: list[Command] = []
    for k in ks:
        argv = list(py_command("eval.py", "--k", k, "-j", jobs, root=root))
        for dataset in datasets:
            argv.extend(("--dataset", dataset))
        if force:
            argv.append("--force")
        commands.append(Command(f"DEBD evaluation epsilon={k}", tuple(argv)))
    return commands


def build_cw_command(
    datasets: Sequence[str],
    ks: Sequence[int],
    *,
    jobs: int,
    root: Path = ROOT,
) -> Command:
    argv = list(
        py_command(
            "learn_mle_adv.py",
            "--k",
            0,
            *ks,
            "-j",
            jobs,
            root=root,
        )
    )
    if datasets:
        argv.append("--datasets")
        argv.extend(datasets)
    return Command("CW auxiliary MLE PCs", tuple(argv))


def run_debd_corrupt(args: argparse.Namespace) -> None:
    datasets = select_datasets(args.datasets)
    ks = select_ks(args.ks)
    if not args.dry_run:
        require_debd_inputs(datasets)
    run_commands(
        build_random_corruption_commands(
            datasets, ks, jobs=args.jobs, force=args.force
        ),
        jobs=1,
        dry_run=args.dry_run,
    )
    run_commands(
        build_mle_attack_commands(datasets, ks, force=args.force),
        jobs=args.jobs,
        dry_run=args.dry_run,
    )


def run_debd_peter(args: argparse.Namespace) -> None:
    datasets = select_datasets(args.datasets)
    ks = select_ks(args.ks)
    if not args.dry_run:
        require_debd_inputs(datasets)
    run_commands(
        build_debd_peter_commands(datasets, ks, force=args.force),
        jobs=args.jobs,
        dry_run=args.dry_run,
    )


def run_debd_rltpm(args: argparse.Namespace) -> None:
    datasets = select_datasets(args.datasets)
    ks = select_ks(args.ks)
    if not args.dry_run:
        require_debd_inputs(datasets)
    run_commands(
        build_rltpm_commands(datasets, ks, jobs=args.jobs),
        jobs=1,
        dry_run=args.dry_run,
    )


def run_debd_attack(args: argparse.Namespace) -> None:
    datasets = select_datasets(args.datasets)
    ks = select_ks(args.ks)
    commands = build_method_attack_commands(
        datasets, ks, force=args.force
    )
    if not args.dry_run:
        require_debd_inputs(datasets)
        missing = [
            command.argv[2]
            for command in commands
            if not Path(command.argv[2]).is_file()
        ]
        if missing:
            raise ReproductionError(
                "Method-attack preflight found missing circuit(s):\n  "
                + "\n  ".join(missing[:20])
            )
    run_commands(commands, jobs=args.jobs, dry_run=args.dry_run)


def run_debd_evaluate(args: argparse.Namespace) -> None:
    datasets = select_datasets(args.datasets)
    ks = select_ks(args.ks)
    run_commands(
        build_eval_commands(
            datasets, ks, jobs=args.jobs, force=args.force
        ),
        jobs=1,
        dry_run=args.dry_run,
    )


def cw_means(
    rows: Sequence[tuple[str, dict[int, float | None]]],
    expected_datasets: Sequence[str],
) -> dict[int, float]:
    by_dataset = {dataset: scores for dataset, scores in rows}
    missing = sorted(set(expected_datasets) - set(by_dataset))
    extra = sorted(set(by_dataset) - set(expected_datasets))
    if missing or extra:
        raise ReproductionError(
            "CW dataset coverage mismatch"
            + (f"; missing: {', '.join(missing)}" if missing else "")
            + (f"; unexpected: {', '.join(extra)}" if extra else "")
        )
    means: dict[int, float] = {}
    for k in EPSILONS:
        undefined = [
            dataset
            for dataset in expected_datasets
            if by_dataset[dataset].get(k) is None
        ]
        if undefined:
            raise ReproductionError(
                f"Undefined Circuit-Wasserstein values for epsilon={k}: "
                + ", ".join(undefined)
            )
        values = [float(by_dataset[dataset][k]) for dataset in expected_datasets]
        means[k] = sum(float(value) for value in values) / len(values)
    return means


def render_cw_artifacts(*, output_dir: Path = OUTPUT_DIR) -> dict[int, float]:
    try:
        import print_distances
    except ImportError as exc:
        raise ReproductionError(
            "Circuit-Wasserstein rendering requires the SparC environment."
        ) from exc
    rows = print_distances.collect_rows(list(EPSILONS))
    if not rows:
        raise ReproductionError(
            "No CW distances were computed. Run `python reproduce.py debd cw` first."
        )
    expected_datasets = debd_dataset_names()
    means = cw_means(rows, expected_datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_p": 1.0,
        "scale_factor": 1.0,
        "rows": {
            dataset: {str(k): values[k] for k in EPSILONS}
            for dataset, values in rows
        },
        "means": {str(k): means[k] for k in EPSILONS},
    }
    (output_dir / "table3.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "% Requires \\usepackage{booktabs}",
        "\\begin{tabular}{cr}",
        "\\toprule",
        "$\\epsilon$ & Mean Circuit-Wasserstein distance \\\\",
        "\\midrule",
        *(f"{k} & {means[k]:.2f} \\\\" for k in EPSILONS),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (output_dir / "table3.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_dir / 'table3.tex'} and table3.json")
    return means


def run_debd_cw(args: argparse.Namespace) -> None:
    datasets = select_datasets(args.datasets)
    ks = select_ks(args.ks)
    if not args.dry_run:
        require_debd_inputs(datasets)
    run_command(
        build_cw_command(datasets, ks, jobs=args.jobs),
        dry_run=args.dry_run,
    )
    report_argv = list(py_command("print_distances.py"))
    for k in ks:
        report_argv.extend(("--k", str(k)))
    full_scope = (
        set(datasets) == set(debd_dataset_names()) and set(ks) == set(EPSILONS)
    )
    if not full_scope:
        run_command(
            Command("CW distance report", tuple(report_argv)),
            dry_run=args.dry_run,
        )
        print(
            "skip  canonical Table 3 rendering: a complete 28-dataset, "
            "epsilon={1,3,5} CW scope is required."
        )
    elif args.dry_run:
        run_command(
            Command("CW distance report and canonical Table 3", tuple(report_argv)),
            dry_run=True,
        )
    elif not args.dry_run:
        render_cw_artifacts()


def run_debd_runtime(args: argparse.Namespace) -> None:
    run_command(
        Command("PeTeR runtime benchmark", py_command("runtime.py")),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"write reviewer hardware provenance -> {OUTPUT_DIR / 'table4_provenance.json'}")
    else:
        write_runtime_provenance()
    run_command(
        Command(
            "runtime table",
            py_command("table2.py", "-o", OUTPUT_DIR / "table4.tex"),
        ),
        dry_run=args.dry_run,
    )


def mnist_winner(*, root: Path = ROOT) -> dict[str, Any]:
    return tpe_winner("mnist", 1, root=root)


def mnist_peter_dir(*, root: Path = ROOT) -> Path:
    winner = mnist_winner(root=root)
    return (
        root
        / "results"
        / "mnist"
        / "k1"
        / hyperparameter_dir(float(winner["lr"]), float(winner["ratio"]))
    )


def mnist_prepare_outputs(*, root: Path = ROOT) -> list[Path]:
    outputs = [
        root / "original_datasets" / "mnist" / "mnist.test.data",
        root / "original_datasets" / "mnist" / "mnist.test.n500.data",
    ]
    for sigma in (0.1, *(index / 1000 for index in range(1, 11))):
        outputs.extend(
            root
            / "corrupted_datasets"
            / "mnist"
            / f"sigma{sigma:.3f}"
            / f"r{replicate}.data"
            for replicate in range(10)
        )
    return outputs


def mnist_prepare_marker(*, root: Path = ROOT) -> Path:
    return root / "corrupted_datasets" / "mnist" / "prepare_complete.json"


def mnist_prepare_complete(*, root: Path = ROOT) -> bool:
    marker = mnist_prepare_marker(root=root)
    if not marker.is_file():
        return False
    try:
        payload = load_json(marker)
    except ReproductionError:
        return False
    expected = mnist_prepare_outputs(root=root)
    expected_names = [path.relative_to(root).as_posix() for path in expected]
    if payload.get("outputs") != expected_names:
        return False
    expected_sizes = payload.get("sizes", {})
    return all(
        path.is_file()
        and path.stat().st_size > 0
        and expected_sizes.get(path.relative_to(root).as_posix())
        == path.stat().st_size
        for path in expected
    )


def write_mnist_prepare_marker(*, root: Path = ROOT) -> Path:
    outputs = mnist_prepare_outputs(root=root)
    missing = [path for path in outputs if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ReproductionError(
            "MNIST preparation completed without all required outputs:\n  "
            + "\n  ".join(str(path) for path in missing[:20])
        )
    marker = mnist_prepare_marker(root=root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "full_test_rows": 10000,
                "peter_rows": 500,
                "copies_per_sigma": 10,
                "outputs": [path.relative_to(root).as_posix() for path in outputs],
                "sizes": {
                    path.relative_to(root).as_posix(): path.stat().st_size
                    for path in outputs
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def run_mnist_prepare(args: argparse.Namespace) -> None:
    marker = mnist_prepare_marker()
    skip = marker if not args.force and mnist_prepare_complete() else None
    if not args.dry_run and skip is None:
        marker.unlink(missing_ok=True)
    result = run_command(
        Command("prepare MNIST data", py_command("prepare_mnist_data.py"), skip),
        dry_run=args.dry_run,
    )
    if result == "completed":
        written = write_mnist_prepare_marker()
        print(f"wrote {written}")


def run_mnist_learn(args: argparse.Namespace) -> None:
    reference = ROOT / "mnist" / "hclt_mnist_blocksize4.json"
    if reference.is_file() and not args.yes_overwrite_reference and not args.dry_run:
        raise ReproductionError(
            "From-scratch learning overwrites the included paper reference PC. "
            "Pass --yes-overwrite-reference to confirm; exact paper values require "
            "restoring the included reference afterward."
        )
    run_command(
        Command("learn MNIST MLE-PC", py_command("learn_mnist_pc.py")),
        dry_run=args.dry_run,
    )


def build_mnist_peter_command(
    *, force: bool = False, root: Path = ROOT
) -> Command:
    winner = mnist_winner(root=root)
    final_iters = int(
        load_manifest("experiments.json", root=root)["mnist"]["peter"][
            "final_iterations"
        ]
    )
    run_dir = mnist_peter_dir(root=root)
    marker = run_dir / "reproduction_complete.json"
    skip_if = marker if not force and mnist_peter_complete(root=root) else None
    argv = list(
        py_command(
            "peter_mnist.py",
            "--k",
            1,
            "--lr",
            repr(float(winner["lr"])),
            "--ratio",
            repr(float(winner["ratio"])),
            "--iters",
            final_iters,
            root=root,
        )
    )
    return Command("MNIST PeTeR final run", tuple(argv), skip_if)


def mnist_peter_complete(*, root: Path = ROOT) -> bool:
    run_dir = mnist_peter_dir(root=root)
    marker = run_dir / "reproduction_complete.json"
    required = (
        run_dir / "circuit.json",
        run_dir / "config.json",
        run_dir / "metrics.json",
    )
    if not marker.is_file() or any(not path.is_file() for path in required):
        return False
    try:
        payload = load_json(marker)
        winner = mnist_winner(root=root)
        final_iters = int(
            load_manifest("experiments.json", root=root)["mnist"]["peter"][
                "final_iterations"
            ]
        )
        return (
            int(payload["iters"]) == final_iters
            and float(payload["lr"]) == float(winner["lr"])
            and float(payload["ratio"]) == float(winner["ratio"])
            and payload["circuit_sha256_normalized_lf"]
            == sha256_normalized_file(run_dir / "circuit.json")
        )
    except (KeyError, ReproductionError, TypeError, ValueError):
        return False


def write_mnist_peter_marker(*, root: Path = ROOT) -> Path:
    run_dir = mnist_peter_dir(root=root)
    winner = mnist_winner(root=root)
    final_iters = int(
        load_manifest("experiments.json", root=root)["mnist"]["peter"][
            "final_iterations"
        ]
    )
    circuit = run_dir / "circuit.json"
    metrics = run_dir / "metrics.json"
    config = run_dir / "config.json"
    missing = [path for path in (circuit, metrics, config) if not path.is_file()]
    if missing:
        raise ReproductionError(
            "MNIST PeTeR did not produce all completion files:\n  "
            + "\n  ".join(str(path) for path in missing)
        )
    if int(load_json(config)["iters"]) != final_iters:
        raise ReproductionError("MNIST PeTeR config does not record 1000 iterations.")
    marker = run_dir / "reproduction_complete.json"
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "iters": final_iters,
                "lr": float(winner["lr"]),
                "ratio": float(winner["ratio"]),
                "circuit_sha256_normalized_lf": sha256_normalized_file(circuit),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def run_mnist_peter(args: argparse.Namespace) -> None:
    if not args.dry_run:
        required = (
            ROOT / "mnist" / "hclt_mnist_blocksize4.json",
            ROOT / "original_datasets" / "mnist" / "mnist.test.n500.data",
            ROOT / "corrupted_datasets" / "mnist" / "sigma0.100" / "r0.data",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise ReproductionError(
                "MNIST PeTeR preflight failed:\n  "
                + "\n  ".join(str(path) for path in missing)
                + "\nRun `python reproduce.py mnist prepare` first."
            )
    command = build_mnist_peter_command(force=args.force)
    if not args.dry_run and command.skip_if is None:
        (mnist_peter_dir() / "reproduction_complete.json").unlink(missing_ok=True)
    result = run_command(
        command,
        dry_run=args.dry_run,
    )
    if result == "completed":
        marker = write_mnist_peter_marker()
        print(f"wrote {marker}")


def run_mnist_attack(args: argparse.Namespace) -> None:
    argv = list(py_command("fgsm_mnist.py", "--k", 1))
    if args.force:
        argv.append("--force")
    if args.batch_size:
        argv.extend(("--batch-size", str(args.batch_size)))
    run_command(Command("MNIST FGSM attacks", tuple(argv)), dry_run=args.dry_run)


def run_mnist_evaluate(args: argparse.Namespace) -> None:
    argv = list(py_command("eval_mnist.py", "--k", 1))
    if args.force:
        argv.append("--force")
    run_command(Command("MNIST evaluation", tuple(argv)), dry_run=args.dry_run)


def render_table1(*, output: Path) -> None:
    try:
        import table
    except ImportError as exc:
        raise ReproductionError("Table 1 rendering requires the SparC environment.") from exc
    expected = load_manifest("paper_results.json")["paper_tables"]["table1"]
    order = expected["dataset_order"]
    rows = {row[0]: row for row in table.collect_rows(int(expected["epsilon"]))}
    missing = [dataset for dataset in order if dataset not in rows]
    if missing:
        raise ReproductionError(
            "Table 1 caches missing for: " + ", ".join(missing)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        table.to_latex([rows[dataset] for dataset in order], int(expected["epsilon"])),
        encoding="utf-8",
    )
    print(f"wrote {output}")


def artifact_commands(*, skip_runtime: bool = False) -> list[Command]:
    expected = load_manifest("paper_results.json")
    table2_order = expected["paper_tables"]["table2"]["dataset_order"]
    table2_argv = list(py_command("table1.py"))
    for dataset in table2_order:
        table2_argv.extend(("--dataset", dataset))
    table2_argv.extend(("-o", str(OUTPUT_DIR / "table2.tex")))
    commands = [
        Command("paper Table 2", tuple(table2_argv)),
        Command(
            "paper Table 5",
            py_command("table1.py", "-o", OUTPUT_DIR / "table5.tex"),
        ),
    ]
    if not skip_runtime:
        commands.append(
            Command(
                "paper Table 4",
                py_command("table2.py", "-o", OUTPUT_DIR / "table4.tex"),
            )
        )
    commands.extend(
        (
            Command(
                "paper Figure 2",
                py_command("plot2.py", "-o", OUTPUT_DIR / "figure2.pdf"),
            ),
            Command(
                "paper Figure 3",
                py_command(
                    "plot1.py",
                    "-o",
                    OUTPUT_DIR / "figure3.pdf",
                    "--output-a",
                    OUTPUT_DIR / "figure3_random.pdf",
                    "--output-b",
                    OUTPUT_DIR / "figure3_adversarial.pdf",
                ),
            ),
            Command(
                "paper Figure 4",
                py_command(
                    "mnist_visualizer.py",
                    "--k",
                    1,
                    "--seed",
                    0,
                    "-o",
                    OUTPUT_DIR / "figure4.png",
                ),
            ),
        )
    )
    return commands


def run_artifacts(args: argparse.Namespace) -> None:
    if args.dry_run:
        print(f"run   paper Table 1\n  -> {OUTPUT_DIR / 'table1.tex'}")
        print(f"run   paper Table 3\n  -> {OUTPUT_DIR / 'table3.tex'}")
    else:
        render_table1(output=OUTPUT_DIR / "table1.tex")
        render_cw_artifacts()
    run_commands(
        artifact_commands(skip_runtime=args.skip_runtime),
        jobs=1,
        dry_run=args.dry_run,
    )
    if not args.skip_runtime and not args.dry_run:
        write_runtime_provenance()


def write_runtime_provenance(*, root: Path = ROOT) -> Path:
    experiments = load_manifest("experiments.json", root=root)
    payload = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "processor": platform.processor() or "unknown",
            "logical_cpus": os.cpu_count(),
        },
        "packages": {
            name: installed_version(name)
            for name in ("sparc-pc", "numpy", "scipy")
        },
        "methodology": experiments["debd"]["runtime"],
        "numeric_equality_to_paper_required": False,
    }
    output = root / "paper_outputs" / "table4_provenance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return output


def _cache_record(path: Path) -> dict[str, float]:
    payload = load_json(path)
    return {
        key: float(payload[key])
        for key in ("orig_test_ll", "own_adv_ll", "rand_mean_ll", "rand_std_ll")
    }


def _actual_debd_cache(
    dataset: str,
    k: int,
    method: str,
    *,
    root: Path = ROOT,
) -> dict[str, float]:
    if method == "mle_pc":
        path = root / "adversarial_datasets" / f"{dataset}_K{k}.eval.json"
    elif method == "rltpm":
        path = (
            root
            / "rltpm_learned_pcs"
            / "hclt"
            / dataset
            / "4"
            / f"K{k}"
            / f"{dataset}_K{k}_rltpm.eval.json"
        )
    elif method == "peter":
        winner = tpe_winner(dataset, k, root=root)
        path = (
            debd_peter_output(dataset, k, winner, root=root)
            / f"{dataset}_K{k}_peter.eval.json"
        )
    else:
        raise ValueError(method)
    return _cache_record(path)


def _round2_record(record: dict[str, float]) -> dict[str, str]:
    return {key: f"{value:.2f}" for key, value in record.items()}


def truncate_toward_zero(value: float, places: int = 0) -> float:
    factor = 10**places
    return int(value * factor) / factor


def rounded_fingerprint(payload: Any, *, places: int = 10) -> str:
    def normalize(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, places)
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    encoded = json.dumps(
        normalize(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _figure_record(
    values: dict[Any, Any], dataset: str, k: int, method: str
) -> dict[str, float]:
    key = (dataset, k, method)
    if key in values:
        return values[key]
    return values[dataset][str(k)][method]


def figure3_source_payload(
    values: dict[Any, Any],
    dimensions: dict[str, int],
    datasets: Sequence[str],
) -> dict[str, list[dict[str, float | str]]]:
    payload: dict[str, list[dict[str, float | str]]] = {}
    for k in EPSILONS:
        rows: list[dict[str, float | str]] = []
        for dataset in datasets:
            variables = dimensions[dataset]
            peter = _figure_record(values, dataset, k, "peter")
            rltpm = _figure_record(values, dataset, k, "rltpm")
            peter_random = peter["rand_mean_ll"] / variables
            rltpm_random = rltpm["rand_mean_ll"] / variables
            peter_adversarial = peter["own_adv_ll"] / variables
            rltpm_adversarial = rltpm["own_adv_ll"] / variables
            rows.append(
                {
                    "dataset": dataset,
                    "peter_random_per_variable": peter_random,
                    "rltpm_random_per_variable": rltpm_random,
                    "random_percent_improvement": (
                        100.0
                        * (peter_random - rltpm_random)
                        / abs(rltpm_random)
                    ),
                    "peter_adversarial_per_variable": peter_adversarial,
                    "rltpm_adversarial_per_variable": rltpm_adversarial,
                    "adversarial_percent_improvement": (
                        100.0
                        * (peter_adversarial - rltpm_adversarial)
                        / abs(rltpm_adversarial)
                    ),
                }
            )
        payload[str(k)] = rows
    return payload


def figure2_aggregate_payload(
    values: dict[Any, Any],
    dimensions: dict[str, int],
    datasets: Sequence[str],
) -> dict[str, dict[str, dict[str, list[float] | int]]]:
    import numpy as np

    payload: dict[str, dict[str, dict[str, list[float] | int]]] = {}
    for panel, metric in (
        ("random", "rand_mean_ll"),
        ("adversarial", "own_adv_ll"),
    ):
        payload[panel] = {}
        for method in ("peter", "rltpm"):
            payload[panel][method] = {}
            for k in EPSILONS:
                rows: list[tuple[float, float]] = []
                for dataset in datasets:
                    variables = dimensions[dataset]
                    mle = _figure_record(values, dataset, k, "mle_pc")
                    robust = _figure_record(values, dataset, k, method)
                    rows.append(
                        (
                            (mle["orig_test_ll"] - robust["orig_test_ll"])
                            / variables,
                            (robust[metric] - mle[metric]) / variables,
                        )
                    )
                xs = np.asarray([row[0] for row in rows], dtype=np.float64)
                ys = np.asarray([row[1] for row in rows], dtype=np.float64)

                def median_ci(data: Any, seed: int) -> list[float]:
                    rng = np.random.default_rng(seed)
                    indices = rng.integers(0, len(data), size=(10000, len(data)))
                    bootstraps = np.median(data[indices], axis=1)
                    low, high = np.percentile(bootstraps, (2.5, 97.5))
                    return [float(np.median(data)), float(low), float(high)]

                payload[panel][method][str(k)] = {
                    "n": len(rows),
                    "x_median_ci": median_ci(xs, k),
                    "y_median_ci": median_ci(ys, 1000 + k),
                }
    return payload


def selected_line_hashes(path: Path, indices: Sequence[int]) -> dict[str, str]:
    lines = normalized_text_bytes(path.read_bytes()).splitlines()
    return {
        str(index): sha256_bytes(lines[index])
        for index in indices
        if 0 <= index < len(lines)
    }


def verification_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    expected: Any = None,
    actual: Any = None,
    detail: str | None = None,
) -> None:
    check = {
        "name": name,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
    }
    if detail:
        check["detail"] = detail
    checks.append(check)
    state = "PASS" if passed else "FAIL"
    print(f"{state}  {name}")


def verify_scientific_hashes(
    checks: list[dict[str, Any]], *, root: Path = ROOT
) -> None:
    manifest = load_manifest("scientific_code.json", root=root)
    for name, expected in manifest["post_cleanup_sha256_normalized_lf"].items():
        path = root / name
        actual = sha256_normalized_file(path) if path.is_file() else None
        verification_check(
            checks,
            f"scientific source hash: {name}",
            actual == expected,
            expected=expected,
            actual=actual,
        )


def verify_paper(
    *, root: Path = ROOT, allow_missing_artifacts: bool = False
) -> dict[str, Any]:
    expected = load_manifest("paper_results.json", root=root)
    checks: list[dict[str, Any]] = []
    current: dict[tuple[str, int, str], dict[str, float]] = {}

    table5 = expected["paper_tables"]["table5"]
    for dataset in table5["dataset_order"]:
        for k in EPSILONS:
            for method in ("mle_pc", "rltpm", "peter"):
                expected_record = table5["values"][dataset][str(k)][method]
                try:
                    actual_record = _actual_debd_cache(dataset, k, method, root=root)
                    current[(dataset, k, method)] = actual_record
                    expected_display = _round2_record(expected_record)
                    actual_display = _round2_record(actual_record)
                    verification_check(
                        checks,
                        f"Table 5 {dataset} epsilon={k} {method}",
                        actual_display == expected_display,
                        expected=expected_display,
                        actual=actual_display,
                    )
                except (ReproductionError, FileNotFoundError, KeyError) as exc:
                    verification_check(
                        checks,
                        f"Table 5 {dataset} epsilon={k} {method}",
                        False,
                        expected=_round2_record(expected_record),
                        actual=None,
                        detail=str(exc),
                    )

    table1 = expected["paper_tables"]["table1"]
    for dataset in table1["dataset_order"]:
        train_cache = (
            root
            / "original_datasets"
            / dataset
            / f"{dataset}.train.mle_train_ll.json"
        )
        try:
            actual_train = float(load_json(train_cache)["train_ll"])
            expected_train = float(table1["values"][dataset]["train_ll"])
            verification_check(
                checks,
                f"Table 1 {dataset} train LL",
                f"{actual_train:.2f}" == f"{expected_train:.2f}",
                expected=f"{expected_train:.2f}",
                actual=f"{actual_train:.2f}",
            )
        except ReproductionError as exc:
            verification_check(
                checks,
                f"Table 1 {dataset} train LL",
                False,
                expected=f"{table1['values'][dataset]['train_ll']:.2f}",
                actual=None,
                detail=str(exc),
            )

    expected_cache_count = len(table5["dataset_order"]) * len(EPSILONS) * 3
    if len(current) == expected_cache_count:
        dimensions = {
            dataset: int(record["variables"])
            for dataset, record in load_manifest("debd.json", root=root)[
                "datasets"
            ].items()
        }
        expected_figure3 = figure3_source_payload(
            table5["values"], dimensions, table5["dataset_order"]
        )
        actual_figure3 = figure3_source_payload(
            current, dimensions, table5["dataset_order"]
        )
        verification_check(
            checks,
            "Figure 3 full-precision source arrays",
            rounded_fingerprint(actual_figure3)
            == rounded_fingerprint(expected_figure3),
            expected=rounded_fingerprint(expected_figure3),
            actual=rounded_fingerprint(actual_figure3),
        )
        expected_figure2 = figure2_aggregate_payload(
            table5["values"], dimensions, table5["dataset_order"]
        )
        actual_figure2 = figure2_aggregate_payload(
            current, dimensions, table5["dataset_order"]
        )
        verification_check(
            checks,
            "Figure 2 medians and bootstrap intervals",
            rounded_fingerprint(actual_figure2)
            == rounded_fingerprint(expected_figure2),
            expected=rounded_fingerprint(expected_figure2),
            actual=rounded_fingerprint(actual_figure2),
        )
    else:
        for figure_name in (
            "Figure 3 full-precision source arrays",
            "Figure 2 medians and bootstrap intervals",
        ):
            verification_check(
                checks,
                figure_name,
                False,
                expected=expected_cache_count,
                actual=len(current),
                detail="Incomplete DEBD evaluation caches.",
            )

    stats_expected = expected["debd_statistics"]["random_corruption"]
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        verification_check(
            checks,
            "DEBD Wilcoxon dependency",
            False,
            expected="scipy",
            actual=None,
            detail=str(exc),
        )
    else:
        for k in EPSILONS:
            peter = [
                current[(dataset, k, "peter")]["rand_mean_ll"]
                for dataset in table5["dataset_order"]
                if (dataset, k, "peter") in current
                and (dataset, k, "rltpm") in current
            ]
            rltpm = [
                current[(dataset, k, "rltpm")]["rand_mean_ll"]
                for dataset in table5["dataset_order"]
                if (dataset, k, "peter") in current
                and (dataset, k, "rltpm") in current
            ]
            expected_test = stats_expected["tests"][str(k)]
            if len(peter) != len(table5["dataset_order"]):
                verification_check(
                    checks,
                    f"DEBD statistics epsilon={k}",
                    False,
                    expected=len(table5["dataset_order"]),
                    actual=len(peter),
                    detail="Incomplete evaluation caches",
                )
                continue
            wins = sum(left > right for left, right in zip(peter, rltpm))
            result = wilcoxon(
                peter,
                rltpm,
                zero_method="wilcox",
                alternative="two-sided",
                method="exact",
            )
            raw_p = float(result.pvalue)
            comparisons = int(stats_expected["multiple_testing"]["comparisons"])
            adjusted_p = min(1.0, raw_p * comparisons)
            raw_expected = float(expected_test["raw_p"])
            adjusted_expected = float(expected_test["bonferroni_p"])
            passed = (
                wins == int(expected_test["peter_wins"])
                and abs(raw_p - raw_expected) <= max(1e-15, abs(raw_expected) * 1e-12)
                and abs(adjusted_p - adjusted_expected)
                <= max(1e-15, abs(adjusted_expected) * 1e-12)
                and adjusted_p
                < float(stats_expected["multiple_testing"]["alpha"])
            )
            verification_check(
                checks,
                f"DEBD Wilcoxon/Bonferroni epsilon={k}",
                passed,
                expected={
                    "wins": expected_test["peter_wins"],
                    "raw_p": raw_expected,
                    "raw_p_rounded": expected_test["raw_p_rounded"],
                    "bonferroni_p": adjusted_expected,
                    "significant": True,
                },
                actual={
                    "wins": wins,
                    "statistic": float(result.statistic),
                    "raw_p": raw_p,
                    "bonferroni_p": adjusted_p,
                },
            )

    cw_path = root / "paper_outputs" / "table3.json"
    if cw_path.is_file():
        cw_actual = load_json(cw_path)["means"]
        for k, expected_mean in expected["paper_tables"]["table3"]["mean_cw"].items():
            actual_mean = float(cw_actual[k])
            verification_check(
                checks,
                f"Table 3 mean CW epsilon={k}",
                f"{actual_mean:.2f}" == f"{float(expected_mean):.2f}",
                expected=f"{float(expected_mean):.2f}",
                actual=f"{actual_mean:.2f}",
            )
    else:
        verification_check(
            checks,
            "Table 3 mean CW",
            allow_missing_artifacts,
            expected=str(cw_path.relative_to(root)),
            actual=None,
            detail="Run `python reproduce.py debd cw`.",
        )

    mnist_expected = expected["mnist"]
    summary: dict[str, Any] = {}
    try:
        summary = load_json(root / mnist_expected["summary_path"])
        submitted = {
            "mle_clean_ll": float(summary["mle"]["orig_test_ll"]),
            "mle_sigma_0_010_ll": float(summary["mle"]["sigma0.010_mean_ll"]),
            "peter_sigma_0_010_ll": float(
                summary["peter"]["sigma0.010_mean_ll"]
            ),
            "peter_clean_ll": float(summary["peter"]["orig_test_ll"]),
            "mle_own_fgsm_ll": float(summary["fgsm"]["mle_on_mle_fgsm"]),
            "peter_own_fgsm_ll": float(
                summary["fgsm"]["peter_on_peter_fgsm"]
            ),
        }
        for name, actual_value in submitted.items():
            expected_value = int(mnist_expected["headline"][name])
            verification_check(
                checks,
                f"MNIST headline {name}",
                int(actual_value) == expected_value,
                expected=expected_value,
                actual=actual_value,
                detail="The manuscript truncates these displayed integers toward zero.",
            )
    except (ReproductionError, KeyError) as exc:
        verification_check(
            checks,
            "MNIST headline values",
            False,
            expected=mnist_expected["headline"],
            actual=None,
            detail=str(exc),
        )

    significance_expected = mnist_expected["significance"]
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon

        target = significance_expected["target"]
        mle_replicates = [float(value) for value in summary["mle_replicate_ll"][target]]
        peter_replicates = [
            float(value) for value in summary["peter_replicate_ll"][target]
        ]
        significance_result = scipy_wilcoxon(
            peter_replicates,
            mle_replicates,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact",
        )
        raw_p = float(significance_result.pvalue)
        comparisons = int(significance_expected["bonferroni_comparisons"])
        bonferroni_p = min(1.0, raw_p * comparisons)
        expected_raw = float(significance_expected["raw_p"])
        expected_adjusted = float(significance_expected["bonferroni_p"])
        passed = (
            abs(raw_p - expected_raw) <= 1e-15
            and abs(bonferroni_p - expected_adjusted) <= 1e-15
            and bonferroni_p < float(significance_expected["alpha"])
        )
        verification_check(
            checks,
            "MNIST sigma=0.010 Wilcoxon/Bonferroni significance",
            passed,
            expected=significance_expected,
            actual={
                "statistic": float(significance_result.statistic),
                "raw_p": raw_p,
                "bonferroni_p": bonferroni_p,
                "significant": bonferroni_p
                < float(significance_expected["alpha"]),
            },
        )
    except (ImportError, KeyError, ReproductionError, ValueError) as exc:
        verification_check(
            checks,
            "MNIST sigma=0.010 Wilcoxon/Bonferroni significance",
            False,
            expected=significance_expected,
            actual=None,
            detail=str(exc),
        )

    try:
        import numpy as np

        original = np.loadtxt(
            root / "original_datasets" / "mnist" / "mnist.test.data",
            delimiter=",",
            dtype=np.int32,
        )
        corrupted = np.loadtxt(
            root
            / "corrupted_datasets"
            / "mnist"
            / "sigma0.010"
            / "r0.data",
            delimiter=",",
            dtype=np.int32,
        )
        mean_change = float(
            np.abs(original.astype(np.float64) - corrupted.astype(np.float64)).mean()
        )
        expected_change = float(
            mnist_expected["headline"]["mean_abs_intensity_change"]
        )
        displayed_change = truncate_toward_zero(mean_change, 2)
        verification_check(
            checks,
            "MNIST sigma=0.010 mean absolute intensity change",
            f"{displayed_change:.2f}" == f"{expected_change:.2f}",
            expected=f"{expected_change:.2f}",
            actual=f"{mean_change:.6f}",
            detail="The manuscript truncates this displayed value toward zero.",
        )
    except (ImportError, OSError, ValueError) as exc:
        verification_check(
            checks,
            "MNIST sigma=0.010 mean absolute intensity change",
            False,
            expected=mnist_expected["headline"]["mean_abs_intensity_change"],
            actual=None,
            detail=str(exc),
        )

    figure4_expected = expected["figures"]["4"]
    figure4_indices = [int(index) for index in figure4_expected["sample_indices"]]
    figure4_paths = {
        "original": root
        / "original_datasets"
        / "mnist"
        / "mnist.test.data",
        "random_sigma0.010_r0": root
        / "corrupted_datasets"
        / "mnist"
        / "sigma0.010"
        / "r0.data",
        "mle_fgsm": root
        / "adversarial_datasets"
        / "K1"
        / "mnist.test.data",
    }
    for label, path in figure4_paths.items():
        try:
            actual_hashes = selected_line_hashes(path, figure4_indices)
            expected_hashes = figure4_expected["row_sha256"][label]
            verification_check(
                checks,
                f"Figure 4 prescribed rows: {label}",
                actual_hashes == expected_hashes,
                expected=expected_hashes,
                actual=actual_hashes,
            )
        except OSError as exc:
            verification_check(
                checks,
                f"Figure 4 prescribed rows: {label}",
                False,
                expected=figure4_expected["row_sha256"][label],
                actual=None,
                detail=str(exc),
            )

    required_artifacts = [
        "table1.tex",
        "table2.tex",
        "table3.tex",
        "table5.tex",
        "figure2.pdf",
        "figure3.pdf",
        "figure4.png",
    ]
    for name in required_artifacts:
        path = root / "paper_outputs" / name
        verification_check(
            checks,
            f"rendered artifact {name}",
            path.is_file() or allow_missing_artifacts,
            expected="present",
            actual="present" if path.is_file() else "missing (allowed)",
        )
    runtime_table = root / "paper_outputs" / "table4.tex"
    runtime_provenance = root / "paper_outputs" / "table4_provenance.json"
    runtime_expected = expected["paper_tables"]["table4"]
    expected_runtime_rows = int(runtime_expected["expected_rows"])
    runtime_records: list[dict[str, Any]] = []
    runtime_errors: list[str] = []
    runtime_method = load_manifest("experiments.json", root=root)["debd"]["runtime"]
    for dataset in table5["dataset_order"]:
        path = root / "runtime_results" / f"{dataset}.json"
        if not path.is_file():
            runtime_errors.append(f"missing {path.relative_to(root)}")
            continue
        try:
            record = load_json(path)
            if record.get("dataset") != dataset:
                runtime_errors.append(f"dataset mismatch in {path.relative_to(root)}")
            if int(record.get("k", -1)) != int(runtime_method["epsilon"]):
                runtime_errors.append(f"epsilon mismatch in {path.relative_to(root)}")
            if int(record.get("warm_start_iters", -1)) != int(
                runtime_method["warm_start_iterations"]
            ):
                runtime_errors.append(
                    f"warm-start mismatch in {path.relative_to(root)}"
                )
            if int(record.get("max_iters", -1)) != int(
                runtime_method["timed_max_iterations"]
            ):
                runtime_errors.append(f"iteration cap mismatch in {path.relative_to(root)}")
            if float(record.get("max_seconds", -1)) != float(
                runtime_method["timed_max_seconds"]
            ):
                runtime_errors.append(f"time cap mismatch in {path.relative_to(root)}")
            if float(record.get("iterations_per_sec", 0)) <= 0:
                runtime_errors.append(
                    f"non-positive throughput in {path.relative_to(root)}"
                )
            runtime_records.append(record)
        except (ReproductionError, TypeError, ValueError) as exc:
            runtime_errors.append(f"{path.relative_to(root)}: {exc}")

    runtime_missing_allowed = (
        allow_missing_artifacts
        and not runtime_table.is_file()
        and not runtime_provenance.is_file()
        and not runtime_records
    )
    runtime_cache_passed = (
        len(runtime_records) == expected_runtime_rows and not runtime_errors
    ) or runtime_missing_allowed
    verification_check(
        checks,
        "Table 4 runtime cache shape and methodology",
        runtime_cache_passed,
        expected={
            "rows": expected_runtime_rows,
            "methodology": runtime_method,
        },
        actual={
            "rows": len(runtime_records),
            "errors": runtime_errors,
            "numeric_equality_exempt": True,
        },
    )

    if runtime_table.is_file():
        table_lines = runtime_table.read_text(encoding="utf-8").splitlines()
        table_datasets = {
            line.split("&", 1)[0].strip()
            for line in table_lines
            if "&" in line
            and line.rstrip().endswith("\\\\")
            and not line.lstrip().startswith("Dataset")
        }
        table_passed = table_datasets == set(table5["dataset_order"])
        table_actual: Any = sorted(table_datasets)
    else:
        table_passed = allow_missing_artifacts
        table_actual = "missing (allowed)" if allow_missing_artifacts else "missing"
    verification_check(
        checks,
        "Table 4 reviewer-hardware artifact",
        table_passed,
        expected=table5["dataset_order"],
        actual=table_actual,
    )

    if runtime_provenance.is_file():
        provenance = load_json(runtime_provenance)
        platform_record = provenance.get("platform", {})
        provenance_passed = (
            provenance.get("methodology") == runtime_method
            and provenance.get("numeric_equality_to_paper_required") is False
            and bool(platform_record.get("os"))
            and bool(platform_record.get("python"))
            and bool(platform_record.get("processor"))
            and int(platform_record.get("logical_cpus") or 0) > 0
        )
        provenance_actual: Any = provenance
    else:
        provenance_passed = allow_missing_artifacts
        provenance_actual = (
            "missing (allowed)" if allow_missing_artifacts else "missing"
        )
    verification_check(
        checks,
        "Table 4 reviewer hardware provenance",
        provenance_passed,
        expected={
            "methodology": runtime_method,
            "hardware_and_software": "non-empty",
            "numeric_equality_required": False,
        },
        actual=provenance_actual,
    )

    verify_scientific_hashes(checks, root=root)
    failures = [check for check in checks if not check["passed"]]
    report = {
        "schema_version": 1,
        "passed": not failures,
        "checks": checks,
        "summary": {
            "passed": len(checks) - len(failures),
            "failed": len(failures),
            "total": len(checks),
        },
        "runtime_numeric_equality_checked": False,
    }
    output = root / "paper_outputs" / "verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nverification: {report['summary']['passed']}/{report['summary']['total']} "
        f"passed; wrote {output.relative_to(root)}"
    )
    return report


def run_verify_paper(args: argparse.Namespace) -> None:
    if args.dry_run:
        print("run   read-only paper claim verification")
        print(f"  -> {OUTPUT_DIR / 'verification.json'}")
        return
    report = verify_paper(allow_missing_artifacts=args.allow_missing_artifacts)
    if not report["passed"]:
        raise ReproductionError(
            f"{report['summary']['failed']} paper verification check(s) failed."
        )


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def doctor(*, strict: bool, profile: str) -> int:
    debd = load_manifest("debd.json")
    experiments = load_manifest("experiments.json")
    print("Platform")
    print(f"  OS: {platform.platform()}")
    print(f"  Python: {platform.python_version()} ({sys.executable})")
    print(f"  CPU: {platform.processor() or 'unknown'}")
    print(f"  logical CPUs: {os.cpu_count() or 'unknown'}")
    print(f"  repository: {ROOT}")

    packages = (
        "sparc-pc",
        "numpy",
        "scipy",
        "matplotlib",
        "optuna",
        "pyjuice",
        "torch",
        "torchvision",
    )
    versions = {name: installed_version(name) for name in packages}
    print("\nPackages")
    for name, version in versions.items():
        print(f"  {name}: {version or 'not installed'}")

    try:
        import torch

        print("\nPyTorch")
        print(f"  CUDA runtime: {torch.version.cuda or 'none'}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
    except ImportError:
        print("\nPyTorch: not installed")

    problems: list[str] = []
    print("\nIncluded inputs")
    pcs = sorted((ROOT / "example_pcs").glob("*.json"))
    print(f"  DEBD MLE-PCs: {len(pcs)}/28")
    if len(pcs) != 28:
        problems.append(f"expected 28 DEBD MLE-PCs, found {len(pcs)}")
    for dataset, record in debd["datasets"].items():
        path = ROOT / "example_pcs" / f"{dataset}.json"
        if (
            not path.is_file()
            or sha256_normalized_file(path)
            != record["mle_pc_sha256_normalized_lf"]
        ):
            problems.append(f"invalid reference PC: {path.relative_to(ROOT)}")
    mnist_pc = ROOT / experiments["mnist"]["mle_pc"]["reference"]
    mnist_ok = (
        mnist_pc.is_file()
        and sha256_normalized_file(mnist_pc)
        == experiments["mnist"]["mle_pc"][
            "reference_sha256_normalized_lf"
        ]
    )
    print(f"  MNIST reference PC: {'ok' if mnist_ok else 'missing/invalid'}")
    if not mnist_ok:
        problems.append("MNIST reference PC is missing or has the wrong hash")

    summaries = list((ROOT / "sweeps" / "tpe").glob("k*/*/study_summary.json"))
    print(f"  TPE summaries: {len(summaries)}/85")
    if len(summaries) != 85:
        problems.append(f"expected 85 TPE summaries, found {len(summaries)}")
    for dataset in debd["datasets"]:
        for k in EPSILONS:
            try:
                tpe_winner(dataset, k)
            except ReproductionError as exc:
                problems.append(str(exc))
    try:
        tpe_winner("mnist", 1)
    except ReproductionError as exc:
        problems.append(str(exc))

    data_root = ROOT / "original_datasets"
    if data_root.is_dir():
        data_errors = validate_debd_data()
        print(
            f"  DEBD local data: {'ok' if not data_errors else f'{len(data_errors)} error(s)'}"
        )
        if strict and profile in ("sparc", "all"):
            problems.extend(data_errors)
    else:
        print("  DEBD local data: not imported")
        if strict and profile in ("sparc", "all"):
            problems.append("DEBD data is not imported")

    required_versions = {}
    if profile in ("sparc", "all"):
        required_versions["sparc-pc"] = "0.6.1"
    if profile in ("pyjuice", "all"):
        required_versions["pyjuice"] = "2.4.3"
    if strict:
        required_packages: set[str] = {"numpy", "scipy", "matplotlib"}
        if profile in ("sparc", "all"):
            required_packages.update(("sparc-pc", "optuna"))
        if profile in ("pyjuice", "all"):
            required_packages.update(("pyjuice", "torch", "torchvision"))
        for package in sorted(required_packages):
            if versions[package] is None:
                problems.append(f"{package} is required but not installed")
        for package, expected in required_versions.items():
            actual = versions[package]
            if actual != expected:
                problems.append(
                    f"{package}: expected {expected} in strict mode, got {actual or 'missing'}"
                )

    if platform.system() != "Linux":
        print(
            "\nNote: SparC stages support Windows/Linux. PyJuice stages were "
            "paper-tested on Linux and are recommended there; no OS lock is applied."
        )
    if problems:
        print("\nProblems")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("\nDoctor checks passed.")
    return 1 if strict and problems else 0


def run_debd_all(args: argparse.Namespace) -> None:
    if args.source is not None:
        import_debd(
            args.source,
            force=args.force,
            dry_run=args.dry_run,
        )
    elif not args.dry_run:
        datasets = select_datasets(args.datasets)
        require_debd_inputs(datasets)
    run_debd_corrupt(args)
    run_debd_peter(args)
    run_debd_rltpm(args)
    run_debd_attack(args)
    run_debd_evaluate(args)
    if not args.skip_cw:
        run_debd_cw(args)
    if not args.skip_runtime:
        run_debd_runtime(args)


def run_mnist_all(args: argparse.Namespace) -> None:
    run_mnist_prepare(args)
    if args.from_scratch:
        run_mnist_learn(args)
    run_mnist_peter(args)
    run_mnist_attack(args)
    run_mnist_evaluate(args)


def add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Only this DEBD dataset (repeatable; default: all 28).",
    )
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        choices=EPSILONS,
        dest="ks",
        help="Only this epsilon (repeatable; default: 1, 3, 5).",
    )


def add_jobs_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Maximum concurrent jobs (default: 1).",
    )


def add_force_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute/overwrite only where the underlying stage supports it.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the paper experiments without changing the submitted "
            "numerical implementations."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests and print commands without running them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Report environment and validate included inputs."
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing inputs or paper framework-version mismatch.",
    )
    doctor_parser.add_argument(
        "--profile",
        choices=("sparc", "pyjuice", "all"),
        default="all",
        help="Environment profile checked in strict mode (default: all).",
    )

    debd_parser = subparsers.add_parser("debd", help="Run DEBD stages.")
    debd_subparsers = debd_parser.add_subparsers(dest="debd_command", required=True)

    import_parser = debd_subparsers.add_parser(
        "import", help="Import and hash-validate a manually downloaded DEBD copy."
    )
    import_parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Extracted DEBD directory, zip, or tar archive.",
    )
    add_force_argument(import_parser)

    corrupt_parser = debd_subparsers.add_parser(
        "corrupt", help="Create random and MLE-PC-targeted corruptions."
    )
    add_scope_arguments(corrupt_parser)
    add_jobs_argument(corrupt_parser)
    add_force_argument(corrupt_parser)

    peter_parser = debd_subparsers.add_parser(
        "peter", help="Materialize the included TPE-winner PeTeR runs."
    )
    add_scope_arguments(peter_parser)
    add_jobs_argument(peter_parser)
    add_force_argument(peter_parser)

    rltpm_parser = debd_subparsers.add_parser(
        "rltpm", help="Learn fixed-configuration RL-TPM circuits."
    )
    add_scope_arguments(rltpm_parser)
    add_jobs_argument(rltpm_parser)

    attack_parser = debd_subparsers.add_parser(
        "attack", help="Attack selected materialized PeTeR/RL-TPM circuits."
    )
    add_scope_arguments(attack_parser)
    add_jobs_argument(attack_parser)
    add_force_argument(attack_parser)

    evaluate_parser = debd_subparsers.add_parser(
        "evaluate", help="Evaluate MLE-PC, PeTeR, and RL-TPM."
    )
    add_scope_arguments(evaluate_parser)
    add_jobs_argument(evaluate_parser)
    add_force_argument(evaluate_parser)

    cw_parser = debd_subparsers.add_parser(
        "cw", help="Fit Appendix-C auxiliary PCs and render Table 3."
    )
    add_scope_arguments(cw_parser)
    add_jobs_argument(cw_parser)

    debd_subparsers.add_parser(
        "runtime", help="Rebenchmark PeTeR and render hardware-dependent Table 4."
    )

    debd_all_parser = debd_subparsers.add_parser(
        "all", help="Run the complete DEBD pipeline in paper order."
    )
    debd_all_parser.add_argument(
        "--source",
        type=Path,
        help="Optional DEBD directory/archive to import before running.",
    )
    add_scope_arguments(debd_all_parser)
    add_jobs_argument(debd_all_parser)
    add_force_argument(debd_all_parser)
    debd_all_parser.add_argument(
        "--skip-cw", action="store_true", help="Skip PyJuice CW auxiliary fits."
    )
    debd_all_parser.add_argument(
        "--skip-runtime", action="store_true", help="Skip runtime rebenchmarking."
    )

    mnist_parser = subparsers.add_parser("mnist", help="Run MNIST stages.")
    mnist_subparsers = mnist_parser.add_subparsers(
        dest="mnist_command", required=True
    )

    mnist_prepare_parser = mnist_subparsers.add_parser(
        "prepare", help="Download/export MNIST and create Gaussian corruptions."
    )
    add_force_argument(mnist_prepare_parser)

    mnist_learn_parser = mnist_subparsers.add_parser(
        "learn", help="Learn MNIST MLE-PC from scratch (not bit-exact)."
    )
    mnist_learn_parser.add_argument(
        "--yes-overwrite-reference",
        action="store_true",
        help="Confirm overwriting the included exact-paper MNIST reference PC.",
    )

    mnist_peter_parser = mnist_subparsers.add_parser(
        "peter", help="Run the final 1000-iteration MNIST PeTeR configuration."
    )
    add_force_argument(mnist_peter_parser)

    mnist_attack_parser = mnist_subparsers.add_parser(
        "attack", help="Run epsilon-1 finite-difference FGSM attacks."
    )
    add_force_argument(mnist_attack_parser)
    mnist_attack_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help="Rows per finite-difference pass (0 = all rows).",
    )

    mnist_evaluate_parser = mnist_subparsers.add_parser(
        "evaluate", help="Evaluate clean/Gaussian/FGSM MNIST likelihoods."
    )
    add_force_argument(mnist_evaluate_parser)

    mnist_all_parser = mnist_subparsers.add_parser(
        "all", help="Run the complete MNIST pipeline using the reference PC."
    )
    add_force_argument(mnist_all_parser)
    mnist_all_parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Also relearn the unseeded MLE-PC before PeTeR.",
    )
    mnist_all_parser.add_argument(
        "--yes-overwrite-reference",
        action="store_true",
        help="Required with --from-scratch; confirms replacing the reference PC.",
    )
    mnist_all_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        metavar="N",
        help="Rows per finite-difference pass (0 = all rows).",
    )

    artifacts_parser = subparsers.add_parser(
        "artifacts", help="Render canonical paper Tables 1-5 and Figures 2-4."
    )
    artifacts_parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Do not require/render hardware-dependent Table 4.",
    )

    verify_parser = subparsers.add_parser(
        "verify-paper", help="Compare generated results with every paper claim."
    )
    verify_parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Verify numeric caches while treating unrendered outputs as diagnostic skips.",
    )
    return parser


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        if args.dry_run:
            print("run   environment and included-input checks")
            return 0
        return doctor(strict=args.strict, profile=args.profile)
    if args.command == "debd":
        if args.debd_command == "import":
            import_debd(
                args.source, force=args.force, dry_run=args.dry_run
            )
        elif args.debd_command == "corrupt":
            run_debd_corrupt(args)
        elif args.debd_command == "peter":
            run_debd_peter(args)
        elif args.debd_command == "rltpm":
            run_debd_rltpm(args)
        elif args.debd_command == "attack":
            run_debd_attack(args)
        elif args.debd_command == "evaluate":
            run_debd_evaluate(args)
        elif args.debd_command == "cw":
            run_debd_cw(args)
        elif args.debd_command == "runtime":
            run_debd_runtime(args)
        elif args.debd_command == "all":
            run_debd_all(args)
        return 0
    if args.command == "mnist":
        if args.mnist_command == "prepare":
            run_mnist_prepare(args)
        elif args.mnist_command == "learn":
            run_mnist_learn(args)
        elif args.mnist_command == "peter":
            run_mnist_peter(args)
        elif args.mnist_command == "attack":
            run_mnist_attack(args)
        elif args.mnist_command == "evaluate":
            run_mnist_evaluate(args)
        elif args.mnist_command == "all":
            run_mnist_all(args)
        return 0
    if args.command == "artifacts":
        run_artifacts(args)
        return 0
    if args.command == "verify-paper":
        run_verify_paper(args)
        return 0
    raise ReproductionError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except (ReproductionError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
