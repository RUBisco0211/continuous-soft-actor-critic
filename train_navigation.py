#!/usr/bin/env python
"""Train CMASAC/BenchMARL navigation runs with wandb logging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCHMARL_DIR = ROOT / "BenchMARL"
VMAS_DIR = ROOT / "VectorizedMultiAgentSimulator"

os.chdir(ROOT)

LOCAL_IMPORT_PATHS = [ROOT, BENCHMARL_DIR, VMAS_DIR]
for path in reversed(LOCAL_IMPORT_PATHS):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _assert_local_package(module_name: str, expected_parent: Path) -> None:
    module = __import__(module_name)
    module_file = Path(module.__file__).resolve()
    expected_parent = expected_parent.resolve()
    if expected_parent not in module_file.parents:
        raise RuntimeError(
            f"Expected local {module_name} from {expected_parent}, "
            f"but imported {module_file}. Check PYTHONPATH / editable installs."
        )


def _enable_headless_rendering() -> None:
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    import pyglet

    pyglet.options["headless"] = True


def _parse_seeds(value: str) -> set[int]:
    seeds = {int(seed.strip()) for seed in value.split(",") if seed.strip()}
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _build_algorithm_configs(selected: str):
    from benchmarl.algorithms import (
        IqlConfig,
        MaddpgConfig,
        MappoConfig,
        MasacConfig,
        QmixConfig,
        TestConfig,
    )

    algorithms = {
        "test": TestConfig,
        "cmasac": TestConfig,
        "masac": MasacConfig,
        "mappo": MappoConfig,
        "maddpg": MaddpgConfig,
        "iql": IqlConfig,
        "qmix": QmixConfig,
    }
    if selected == "all":
        order = ["qmix", "iql", "mappo", "masac", "maddpg", "test"]
        return [algorithms[name].get_from_yaml() for name in order]
    return [algorithms[selected].get_from_yaml()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VMAS navigation with BenchMARL, wandb metrics, eval videos, and local checkpoints."
    )
    parser.add_argument(
        "--algorithm",
        choices=["test", "cmasac", "masac", "mappo", "maddpg", "iql", "qmix", "all"],
        default="test",
        help="Algorithm to train. 'test' is the author-provided CMASAC config.",
    )
    parser.add_argument("--seeds", type=_parse_seeds, default={0})
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=120_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--project", default="cmasac")
    parser.add_argument("--entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--save-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--keep-checkpoints", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", help="Disable eval rendering/video upload.")
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Use deterministic actions during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_video:
        _enable_headless_rendering()

    from benchmarl.benchmark import Benchmark
    from benchmarl.environments import VmasTask
    from benchmarl.experiment import ExperimentConfig
    from benchmarl.models.mlp import MlpConfig
    import vmas

    _assert_local_package("benchmarl", BENCHMARL_DIR)
    _assert_local_package("vmas", VMAS_DIR)

    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.save_folder = save_dir
    experiment_config.loggers = ["wandb"]
    experiment_config.project_name = args.project
    experiment_config.wandb_extra_kwargs = {
        "mode": args.wandb_mode,
        "tags": ["cmasac", "benchmarl", "vmas", "navigation", args.algorithm],
    }
    if args.entity:
        experiment_config.wandb_extra_kwargs["entity"] = args.entity

    experiment_config.max_n_iters = args.max_iters
    experiment_config.max_n_frames = None
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = args.eval_interval
    experiment_config.evaluation_episodes = args.eval_episodes
    experiment_config.evaluation_deterministic_actions = args.deterministic_eval
    experiment_config.render = not args.no_video
    experiment_config.create_json = True
    experiment_config.checkpoint_interval = args.eval_interval
    experiment_config.checkpoint_at_end = True
    experiment_config.keep_checkpoints_num = (
        None if args.keep_checkpoints == 0 else args.keep_checkpoints
    )
    experiment_config.exclude_buffer_from_checkpoint = False

    benchmark = Benchmark(
        algorithm_configs=_build_algorithm_configs(args.algorithm),
        tasks=[VmasTask.NAVIGATION.get_from_yaml()],
        seeds=args.seeds,
        experiment_config=experiment_config,
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
    )

    for index, experiment in enumerate(benchmark.get_experiments(), start=1):
        print(f"\nRunning experiment {index}/{benchmark.n_experiments}: {experiment.name}\n")
        experiment.run()


if __name__ == "__main__":
    main()
