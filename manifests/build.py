"""Build the checked-in reproduction manifests from the submitted-run artifacts.

This is a maintainer utility, not a reviewer workflow.  The generated manifests
remain usable after the large experiment artifacts are removed from version
control.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests"
EPSILONS = (1, 3, 5)

TABLE1_DATASETS = (
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
)
TABLE2_DATASETS = (
    "baudio",
    "c20ng",
    "connect4",
    "cr52",
    "kdd",
    "kosarek",
    "msnbc",
    "nltcs",
)

# Captured before the cleanup.  The two evaluation files already contained the
# submitted-run reporting additions.  learn_mle_adv.py is the sole allowed
# executable repair (four imports restored).
BASELINE_SCIENTIFIC_HASHES = {
    "robustify.py": "f19e07077fa07f6dbc4fd9096d120ccf7eb5870fa1181685bfca7275a4e722bc",
    "peter.py": "646c61ca75b335b06e0465b2a31aca2916cbbf5691b7fb4829cc92ad91440c3b",
    "peter_mnist.py": "dab0378f281cef08c1be355c084ea3f54ed0dc7c4d27bc2afeb5813f37a5aa01",
    "learn_rltpm.py": "f5522c7a977840732f0c7899d92093583265502467be14b92ed047c6770c04fb",
    "learn_mnist_pc.py": "457908322f315dcb543f6ca22e56e25121c66bc702451040efefef3255795e1b",
    "attack.py": "69e2f2bce936b16112ce8088eca126d37c7d11e6a59791dc112573e8d0bf1ff8",
    "attack_all.py": "a390f956b72da843874a73617769e9f143ae9c0c090f1b0f19d5ce6654f6683d",
    "random_corrupt.py": "c92cffa65672cdd40fa2a4dc6c748d88a9a695f303d3e7640a050a796d408f99",
    "fgsm.py": "96ad25259dbe8d8386544be08278dc5fd6b7183bbc3f0bfe31229bdafb6933fe",
    "eval.py": "2aac9bbb0c9212f8ad393e5f0c3e752abb9e1b547ba36709b94a289309cc0a3e",
    "eval_mnist.py": "4cae5f309cf8e6285bd4bb39e113deff38fd6a2d17c743f6fa86811899940a5d",
    "learn_mle_adv.py": "0f637308e1de43a8527e4b9c5f1843ed7741273b09f9636da4b625238bf0d4a3",
    "prepare_mnist_data.py": "7f629792d41285a0efa13128d4d84f7ac0b24f7151c6aa8b982e3604f9c41fa5",
    "fgsm_mnist.py": "7ba867054dd2e23a166ed59d5f222921274136df769856a011d1f30c0d6e21ab",
    "robustify_mnist.py": "fc0c1d1362b3b942eca334605ee7484446d65104eab8190d178170ffa330bd67",
    "tune.py": "e3366d3e1f2158b45e71e2d6f3956f2a63fe2aaebafa1ee9e3b5152728f1c3f9",
    "tune_mnist.py": "c4c8d3efc53038744e0595646841198426160b160fcc658de84be34235966694",
    "sweep.py": "1fc5e2d46a064bc9e58b6beee9c95e5d0fbd7eeb7f7e5faa418db2bdbc05b849",
    "sweep_mnist.py": "fbd93cfce9f15cfde9928f802a599d54a43c797fabffb995be82c60fe5c6e4c6",
    "best_sweep.py": "f5883f8edf3e29556df1a0ab690bed25003f4a2dbd3fbf8fc647d6fd262a50d9",
    "sweep_io.py": "6cc43faea69c93bbcc2477842bb1cda22b5b2c08e3ef4efd47f3cf38f9e8a551",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, payload: dict[str, Any]) -> None:
    path = MANIFEST_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_text_bytes(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def normalized_file_hash(path: Path) -> str:
    return sha256_bytes(normalized_text_bytes(path.read_bytes()))


def selected_line_hashes(path: Path, indices: list[int]) -> dict[str, str]:
    wanted = set(indices)
    hashes: dict[str, str] = {}
    for index, line in enumerate(normalized_text_bytes(path.read_bytes()).splitlines()):
        if index in wanted:
            hashes[str(index)] = sha256_bytes(line)
    if set(map(int, hashes)) != wanted:
        raise ValueError(f"Missing prescribed rows in {path}")
    return hashes


def data_file_record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    normalized = normalized_text_bytes(raw)
    lines = [line for line in normalized.split(b"\n") if line]
    if not lines:
        raise ValueError(f"Empty dataset: {path}")
    variables = len(lines[0].split(b","))
    if any(len(line.split(b",")) != variables for line in lines):
        raise ValueError(f"Inconsistent row width: {path}")
    return {
        "file": path.name,
        "rows": len(lines),
        "variables": variables,
        "sha256": sha256_bytes(raw),
        "sha256_normalized_lf": sha256_bytes(normalized),
    }


def dataset_names() -> list[str]:
    names = sorted(path.stem for path in (ROOT / "example_pcs").glob("*.json"))
    if len(names) != 28:
        raise ValueError(f"Expected 28 DEBD MLE PCs, found {len(names)}")
    return names


def build_debd_manifest(names: list[str]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in names:
        split_records = {}
        for split in ("train", "valid", "test"):
            path = ROOT / "original_datasets" / dataset / f"{dataset}.{split}.data"
            if not path.is_file():
                raise FileNotFoundError(path)
            split_records[split] = data_file_record(path)
        widths = {record["variables"] for record in split_records.values()}
        if len(widths) != 1:
            raise ValueError(f"Split width mismatch for {dataset}")
        datasets[dataset] = {
            "variables": widths.pop(),
            "splits": split_records,
            "mle_pc_sha256_normalized_lf": normalized_file_hash(
                ROOT / "example_pcs" / f"{dataset}.json"
            ),
        }
    return {
        "schema_version": 1,
        "source": {
            "name": "Density Estimation Benchmark Datasets",
            "url": "https://github.com/UCLA-StarAI/Density-Estimation-Datasets",
            "download_policy": "Download manually, then pass the extracted directory or archive to reproduce.py.",
        },
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def tpe_summary(dataset: str, k: int) -> dict[str, Any]:
    path = ROOT / "sweeps" / "tpe" / f"k{k}" / dataset / "study_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_json(path)


def winner_record(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best")
    if not isinstance(best, dict):
        raise ValueError(
            f"Missing best trial for {summary.get('dataset')} k={summary.get('k')}"
        )
    return {
        "lr": float(best["lr"]),
        "ratio": float(best["ratio"]),
        "final_adv_test_ll": float(best["final_adv_test_ll"]),
        "trial_number": int(best["trial_number"]),
        "iters": int(summary["iters"]),
        "n_trials_target": int(summary["n_trials_target"]),
        "n_complete": int(summary["n_complete"]),
        "n_failed": int(summary["n_failed"]),
        "summary": (
            Path("sweeps")
            / "tpe"
            / f"k{summary['k']}"
            / str(summary["dataset"])
            / "study_summary.json"
        ).as_posix(),
    }


def build_experiment_manifest(names: list[str]) -> dict[str, Any]:
    debd_winners = {
        dataset: {
            str(k): winner_record(tpe_summary(dataset, k)) for k in EPSILONS
        }
        for dataset in names
    }
    mnist_summary = tpe_summary("mnist", 1)
    return {
        "schema_version": 1,
        "paper_scope": {
            "debd_datasets": names,
            "epsilons": list(EPSILONS),
            "mnist_epsilon": 1,
        },
        "debd": {
            "mle_pc": {
                "structure": "Hidden Chow-Liu Tree",
                "implementation": "PyJuice 2.4.3",
                "parameter_learning": "1000 EM iterations on the training split",
                "included_under": "example_pcs/",
            },
            "random_corruption": {
                "copies": 10,
                "flips": "epsilon bit indices sampled with replacement per row",
                "seed": "first 32 bits of MD5('<dataset>|K<epsilon>|r<replicate>')",
            },
            "adversarial_attack": {
                "algorithm": "greedy model-aware bit flip",
                "budget": "at most epsilon flips per row",
                "candidate_row_cap": 65536,
            },
            "peter": {
                "iterations": 500,
                "warm_start_iterations": 100,
                "theta_samples": 100,
                "eval_every": 1,
                "deterministic": True,
                "eta_lambda": 10.0,
                "cw_metric_p": 1.0,
                "cw_scale_factor": 1.0,
                "tpe": {
                    "lr": {"low": 1e-8, "high": 0.5, "log": True},
                    "ratio": {"low": 0.1, "high": 100.0, "log": True},
                    "startup_trials": 10,
                    "objective": "final_adv_test_ll",
                    "objective_data": "MLE-PC adversarial test set",
                },
                "winners": debd_winners,
            },
            "rltpm": {
                "structure": "Hidden Chow-Liu Tree",
                "block_size": 4,
                "seed": 0,
                "epochs": 1000,
                "batch_size": 512,
                "optimizer": {
                    "method": "EM",
                    "lr": 0.1,
                    "pseudocount": 0.1,
                },
                "scheduler": {
                    "method": "multi_linear",
                    "lrs": [0.9, 0.1, 0.05],
                    "epoch_milestones": [0, 100, 500],
                },
                "max_candidate_rows": 65536,
            },
            "cw_auxiliary": {
                "fit_k": [0, 1, 3, 5],
                "structure": "Hidden Chow-Liu Tree",
                "block_size": 4,
                "seed": 0,
                "epochs": 1000,
                "batch_size": 512,
                "optimizer": {
                    "method": "EM",
                    "lr": 0.1,
                    "pseudocount": 0.1,
                },
                "scheduler": {
                    "method": "multi_linear",
                    "lrs": [0.9, 0.1, 0.05],
                    "epoch_milestones": [0, 100, 500],
                },
                "k0_data": "clean test split",
                "positive_k_data": "PeTeR model-aware adversarial test set",
                "metric_p": 1.0,
                "scale_factor": 1.0,
            },
            "runtime": {
                "epsilon": 1,
                "warm_start_iterations": 100,
                "timed_max_iterations": 250,
                "timed_max_seconds": 30.0,
                "reported_iterations": 500,
            },
        },
        "mnist": {
            "mle_pc": {
                "reference": "mnist/hclt_mnist_blocksize4.json",
                "reference_sha256_normalized_lf": normalized_file_hash(
                    ROOT / "mnist" / "hclt_mnist_blocksize4.json"
                ),
                "structure": "Hidden Chow-Liu Tree",
                "num_latents": 4,
                "categories_per_pixel": 256,
                "epochs": 350,
                "batch_size": 512,
                "optimizer": {
                    "method": "EM",
                    "lr": 0.1,
                    "pseudocount": 0.1,
                },
                "scheduler": {
                    "method": "multi_linear",
                    "lrs": [0.9, 0.1, 0.05],
                    "epoch_milestones": [0, 100, 350],
                },
                "explicit_seed": None,
            },
            "data": {
                "full_test_rows": 10000,
                "peter_rows": 500,
                "tune_sigma": 0.1,
                "evaluation_sigmas": [
                    round(0.001 * index, 3) for index in range(1, 11)
                ],
                "corruption_copies": 10,
                "pixel_formula": "clip(round(x + 255 * Normal(0, sigma^2)), 0, 255)",
            },
            "peter": {
                "epsilon": 1,
                "sweep_iterations": 500,
                "final_iterations": 1000,
                "winner": winner_record(mnist_summary),
            },
            "attack": {
                "epsilon": 1,
                "method": "one-step central-finite-difference sign attack",
                "rows": 500,
                "max_candidate_rows": 65536,
            },
        },
    }


def result_cache(path: Path) -> dict[str, float]:
    payload = read_json(path)
    return {
        key: float(payload[key])
        for key in ("orig_test_ll", "own_adv_ll", "rand_mean_ll", "rand_std_ll")
    }


def peter_result_dir(dataset: str, k: int, winner: dict[str, Any]) -> Path:
    hp = f"lr={winner['lr']:g}_ratio={winner['ratio']:g}"
    return ROOT / "results" / dataset / f"k{k}" / hp


def build_paper_results(
    names: list[str], experiments: dict[str, Any]
) -> dict[str, Any]:
    table5: dict[str, Any] = {}
    for dataset in names:
        table5[dataset] = {}
        for k in EPSILONS:
            winner = experiments["debd"]["peter"]["winners"][dataset][str(k)]
            peter_dir = peter_result_dir(dataset, k, winner)
            table5[dataset][str(k)] = {
                "mle_pc": result_cache(
                    ROOT
                    / "adversarial_datasets"
                    / f"{dataset}_K{k}.eval.json"
                ),
                "rltpm": result_cache(
                    ROOT
                    / "rltpm_learned_pcs"
                    / "hclt"
                    / dataset
                    / "4"
                    / f"K{k}"
                    / f"{dataset}_K{k}_rltpm.eval.json"
                ),
                "peter": result_cache(
                    peter_dir / f"{dataset}_K{k}_peter.eval.json"
                ),
            }

    table1: dict[str, Any] = {}
    for dataset in TABLE1_DATASETS:
        train_cache = (
            ROOT
            / "original_datasets"
            / dataset
            / f"{dataset}.train.mle_train_ll.json"
        )
        table1[dataset] = {
            "train_ll": float(read_json(train_cache)["train_ll"]),
            "test_ll": table5[dataset]["5"]["mle_pc"]["orig_test_ll"],
            "mle_perturbed_ll": table5[dataset]["5"]["mle_pc"]["own_adv_ll"],
            "peter_perturbed_ll": table5[dataset]["5"]["peter"]["own_adv_ll"],
        }

    mnist_summary = read_json(
        ROOT
        / "results"
        / "mnist"
        / "k1"
        / "lr=2.70019e-06_ratio=0.172305"
        / "eval_summary.json"
    )
    figure4_indices = [134, 255, 317, 422]
    figure4_sources = {
        "original": ROOT / "original_datasets" / "mnist" / "mnist.test.data",
        "random_sigma0.010_r0": (
            ROOT
            / "corrupted_datasets"
            / "mnist"
            / "sigma0.010"
            / "r0.data"
        ),
        "mle_fgsm": ROOT / "adversarial_datasets" / "K1" / "mnist.test.data",
    }
    return {
        "schema_version": 1,
        "comparison_policy": {
            "tables_1_2_5": "formatted values must match at two decimal places",
            "table_3": "means must match at two decimal places",
            "mnist": (
                "the manuscript's displayed likelihood integers and 1.17 pixel "
                "change truncate toward zero rather than round to nearest"
            ),
            "table_4": "hardware-dependent; verify procedure and output shape only",
        },
        "paper_tables": {
            "table1": {
                "epsilon": 5,
                "dataset_order": list(TABLE1_DATASETS),
                "values": table1,
            },
            "table2": {"dataset_order": list(TABLE2_DATASETS)},
            "table3": {
                "mean_cw": {"1": 12.08, "3": 17.66, "5": 21.37}
            },
            "table4": {
                "numeric_comparison": False,
                "paper_hardware": "Intel i9-13950HX CPU",
                "expected_rows": 28,
            },
            "table5": {"dataset_order": names, "values": table5},
        },
        "debd_statistics": {
            "random_corruption": {
                "unit": "paired dataset-level mean over 10 corruptions",
                "tests": {
                    "1": {
                        "peter_wins": 27,
                        "raw_p": 2.2351741790771484e-8,
                        "raw_p_rounded": 2e-8,
                        "bonferroni_p": 6.705522537231445e-8,
                    },
                    "3": {
                        "peter_wins": 23,
                        "raw_p": 0.003160528838634491,
                        "raw_p_rounded": 3e-3,
                        "bonferroni_p": 0.009481586515903473,
                    },
                    "5": {
                        "peter_wins": 20,
                        "raw_p": 0.01362369954586029,
                        "raw_p_rounded": 1e-2,
                        "bonferroni_p": 0.04087109863758087,
                    },
                },
                "multiple_testing": {
                    "method": "Bonferroni",
                    "comparisons": 3,
                    "alpha": 0.05,
                },
            }
        },
        "mnist": {
            "summary_path": (
                "results/mnist/k1/lr=2.70019e-06_ratio=0.172305/"
                "eval_summary.json"
            ),
            "headline": {
                "mle_clean_ll": -724,
                "mle_sigma_0_010_ll": -3287,
                "peter_sigma_0_010_ll": -2993,
                "peter_clean_ll": -1423,
                "mle_own_fgsm_ll": -4734,
                "peter_own_fgsm_ll": -4663,
                "mean_abs_intensity_change": 1.17,
            },
            "submitted_values": {
                "mle_clean_ll": float(mnist_summary["mle"]["orig_test_ll"]),
                "mle_sigma_0_010_ll": float(
                    mnist_summary["mle"]["sigma0.010_mean_ll"]
                ),
                "peter_sigma_0_010_ll": float(
                    mnist_summary["peter"]["sigma0.010_mean_ll"]
                ),
                "peter_clean_ll": float(mnist_summary["peter"]["orig_test_ll"]),
                "mle_own_fgsm_ll": float(
                    mnist_summary["fgsm"]["mle_on_mle_fgsm"]
                ),
                "peter_own_fgsm_ll": float(
                    mnist_summary["fgsm"]["peter_on_peter_fgsm"]
                ),
            },
            "significance": {
                "target": "sigma0.010",
                "unit": "paired mean LL over each of 10 Gaussian replicates",
                "test": "two-sided exact Wilcoxon signed-rank",
                "raw_p": 0.001953125,
                "bonferroni_comparisons": 10,
                "bonferroni_p": 0.01953125,
                "alpha": 0.05,
                "significant": True,
            },
        },
        "figures": {
            "2": {
                "producer": "plot2.py",
                "output": "paper_outputs/figure2.pdf",
                "source": "full-precision Table 5 caches and DEBD dimensions",
            },
            "3": {
                "producer": "plot1.py",
                "output": "paper_outputs/figure3.pdf",
                "source": "full-precision Table 5 caches and DEBD dimensions",
            },
            "4": {
                "producer": "mnist_visualizer.py",
                "output": "paper_outputs/figure4.png",
                "seed": 0,
                "shared_rows": 500,
                "sample_indices": figure4_indices,
                "row_sha256": {
                    label: selected_line_hashes(path, figure4_indices)
                    for label, path in figure4_sources.items()
                },
            },
        },
    }


def build_provenance() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "paper_environment": {
            "peter": {
                "os": "Rocky Linux 8.10",
                "cpu": "2 x AMD EPYC 7713",
                "ram": "2 TB DDR4",
                "gpu": None,
                "parallel_cpu_cores": 128,
                "sparc_pc": "0.6.1",
            },
            "pyjuice_baselines": {
                "gpu": "2 x NVIDIA L40S",
                "pyjuice": "2.4.3",
            },
            "runtime_table": {"cpu": "Intel i9-13950HX"},
        },
        "repo_authoritative_execution": {
            "debd_tuning_data": "MLE-PC adversarial test set, not validation",
            "debd_lr_upper_bound": 0.5,
            "rltpm_selection": "single fixed configuration; no sweep in this repo",
            "mnist_tuning": "sigma=0.100 on first 500 test images",
            "mnist_gaussian_evaluation": (
                "sigma=0.001..0.010 on 10000 images; clean and FGSM headline "
                "scores use the 500-image subset"
            ),
            "mnist_multiple_testing_in_eval_cache": "Holm-Bonferroni over 10 sigmas",
            "paper_verifier_additional_test": (
                "plain Bonferroni is computed for the paper's stated claims"
            ),
            "mnist_display_policy": (
                "manuscript headline values truncate toward zero (for example, "
                "-4734.946 is displayed as -4734 and 1.175443 as 1.17)"
            ),
        },
        "manuscript_differences": [
            {
                "topic": "DEBD tuning split",
                "paper": "corrupted validation data disjoint from test",
                "executed": "MLE-adversarial test data",
            },
            {
                "topic": "DEBD TPE learning-rate upper bound",
                "paper": 0.1,
                "executed": 0.5,
            },
            {
                "topic": "RL-TPM tuning",
                "paper": "same validation-based tuning protocol",
                "executed": "fixed settings in learn_rltpm.py",
            },
            {
                "topic": "MNIST data sizes",
                "paper": "not distinguished",
                "executed": "500-image clean/FGSM and 10000-image Gaussian evaluation",
            },
        ],
    }


def build_scientific_hash_manifest() -> dict[str, Any]:
    current = {
        name: normalized_file_hash(ROOT / name)
        for name in BASELINE_SCIENTIFIC_HASHES
    }
    changed = [
        name
        for name, old_hash in BASELINE_SCIENTIFIC_HASHES.items()
        if current[name] != old_hash
    ]
    if changed != ["learn_mle_adv.py"]:
        raise ValueError(f"Unexpected scientific source changes: {changed}")
    return {
        "schema_version": 1,
        "purpose": "Guard the submitted numerical implementation from cleanup edits.",
        "hash_policy": "SHA-256 after CRLF/CR normalization to LF",
        "protected_files": list(BASELINE_SCIENTIFIC_HASHES),
        "baseline_sha256_normalized_lf": BASELINE_SCIENTIFIC_HASHES,
        "post_cleanup_sha256_normalized_lf": current,
        "allowed_change": {
            "learn_mle_adv.py": "Restored pyjuice/torch/DataLoader/serialize_circuit imports only."
        },
    }


def main() -> None:
    names = dataset_names()
    debd = build_debd_manifest(names)
    experiments = build_experiment_manifest(names)
    paper_results = build_paper_results(names, experiments)
    provenance = build_provenance()
    scientific_hashes = build_scientific_hash_manifest()
    write_json("debd.json", debd)
    write_json("experiments.json", experiments)
    write_json("paper_results.json", paper_results)
    write_json("provenance.json", provenance)
    write_json("scientific_code.json", scientific_hashes)


if __name__ == "__main__":
    main()
