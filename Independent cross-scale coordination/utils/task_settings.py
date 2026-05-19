"""
Per-episode task configuration: adaptive task mixture, category labels, pipe geometry, and curriculum.

Covers (1) reweighted sampling over task_a…task_f and release targets,
(2) mapping env ``info`` and category success rules for logged histories,
(3) pipe control points for ``reset`` and straight→curved promotion from category performance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .training_storage import TASK_CATEGORY_KEYS, save_use_curved_pipe_flag


@dataclass
class DynamicSamplingConfig:
    """Hyperparameters for difficulty-based task-mixture and release-target updates."""
    min_task_samples: int = 20
    adjustment_window: int = 50
    adjustment_strength: float = 0.3
    min_probability: float = 0.05
    max_probability: float = 0.9


def compute_dynamic_task_distributions(
    cat_completion_history: dict,
    task_distribution: dict,
    release_distribution: dict,
    cfg: DynamicSamplingConfig,
) -> tuple[dict, dict, dict, dict]:
    """
    Difficulty-weighted update of task-mixture and release-target sampling.
    Returns (new_task_distribution, group_rates, new_release_distribution, release_rates).
    """
    task_rates: dict[str, float] = {}
    for task_key in TASK_CATEGORY_KEYS:
        history = cat_completion_history[task_key]
        if len(history) >= cfg.min_task_samples:
            recent = history[-min(len(history), cfg.adjustment_window) :]
            task_rates[task_key] = float(np.mean(recent))
        else:
            task_rates[task_key] = 0.5

    group_rates = {
        "millicore_only": float(np.mean([task_rates["task_a"], task_rates["task_b"]])),
        "nano_only": task_rates["task_c"],
        "both_exists": float(np.mean([task_rates["task_e"], task_rates["task_f"]])),
    }

    difficulties = {k: 1.0 - v for k, v in group_rates.items()}
    total_difficulty = sum(difficulties.values()) + 1e-6
    new_distribution = {k: v / total_difficulty for k, v in difficulties.items()}

    adjusted_distribution: dict[str, float] = {}
    for key in task_distribution.keys():
        new_prob = (
            new_distribution[key] * cfg.adjustment_strength
            + task_distribution[key] * (1 - cfg.adjustment_strength)
        )
        adjusted_distribution[key] = float(np.clip(new_prob, cfg.min_probability, cfg.max_probability))

    total = sum(adjusted_distribution.values())
    adjusted_distribution = {k: v / total for k, v in adjusted_distribution.items()}

    release_0_rate = float(np.mean([task_rates["task_a"], task_rates["task_e"]]))
    release_1_rate = float(np.mean([task_rates["task_b"], task_rates["task_f"]]))
    release_rates = {0: release_0_rate, 1: release_1_rate}

    release_difficulties = {0: 1.0 - release_0_rate, 1: 1.0 - release_1_rate}
    total_rd = sum(release_difficulties.values()) + 1e-6
    new_release_dist = {k: v / total_rd for k, v in release_difficulties.items()}

    adjusted_release_dist: dict[int, float] = {}
    for key in release_distribution.keys():
        new_prob = (
            new_release_dist[key] * cfg.adjustment_strength
            + release_distribution[key] * (1 - cfg.adjustment_strength)
        )
        adjusted_release_dist[key] = float(np.clip(new_prob, 0.2, 0.8))

    total_r = sum(adjusted_release_dist.values())
    adjusted_release_dist = {k: v / total_r for k, v in adjusted_release_dist.items()}

    return adjusted_distribution, group_rates, adjusted_release_dist, release_rates


def print_dynamic_sampling_update(
    episode_i: int,
    task_distribution: dict,
    group_rates: dict,
    release_distribution: dict,
    release_rates: dict,
) -> None:
    print(f"\nDynamic task distribution update (episode {episode_i}):")
    print(
        f'  Millicore only (task_a/b): completion rate={group_rates["millicore_only"]:.2%}, '
        f'sampling probability={task_distribution["millicore_only"]:.2%}'
    )
    print(
        f'  Nanounits only (task_c): completion rate={group_rates["nano_only"]:.2%}, '
        f'sampling probability={task_distribution["nano_only"]:.2%}'
    )
    print(
        f'  Both Exists (task_e/f): completion rate={group_rates["both_exists"]:.2%}, '
        f'sampling probability={task_distribution["both_exists"]:.2%}'
    )
    print(
        f'  Release=0: completion rate={release_rates[0]:.2%}, '
        f'sampling probability={release_distribution[0]:.2%}'
    )
    print(
        f'  Release=1: completion rate={release_rates[1]:.2%}, '
        f'sampling probability={release_distribution[1]:.2%}'
    )


def infer_task_category(
    millicore_exists: bool, nano_exists: bool, target_release_amount: int
) -> str | None:
    """Map sampled episode configuration to task_a … task_f label."""
    if millicore_exists and not nano_exists:
        return "task_a" if target_release_amount == 0 else "task_b"
    if not millicore_exists and nano_exists:
        return "task_c"
    if millicore_exists and nano_exists:
        return "task_e" if target_release_amount == 0 else "task_f"
    return None


def category_task_completed(
    task_category: str,
    episode_millicore_completed: bool,
    episode_nano_completed: bool,
    episode_release_completed: bool,
) -> bool:
    """Whether the episode satisfies success for that category's active subtasks."""
    if task_category in ("task_a", "task_b"):
        return episode_millicore_completed and episode_release_completed
    if task_category == "task_c":
        return episode_nano_completed
    if task_category in ("task_e", "task_f"):
        return (
            episode_millicore_completed
            and episode_nano_completed
            and episode_release_completed
        )
    return False


def apply_info_to_episode_flags(info: dict) -> tuple[bool, bool, bool, bool, object, object, object]:
    """Map env ``info`` to completion flags and errors. Errors may be None when a subtask is inactive."""
    return (
        info["millicore_reached"],
        info["nano_reached"],
        info["release_correct"],
        info["all_tasks_completed"],
        info.get("millicore_error"),
        info.get("nano_error"),
        info.get("release_error"),
    )


def sample_pipe_control_points(use_curved: bool) -> np.ndarray:
    """Control points for ``simEnv.reset(..., control_points=...)``."""
    if use_curved:
        return np.array(
            [
                [0, random.uniform(0.3, 0.7)],
                [0.25, random.uniform(0.3, 0.7)],
                [0.75, random.uniform(0.3, 0.7)],
                [1, random.uniform(0.3, 0.7)],
            ]
        )
    return np.array(
        [
            [0, 0.5],
            [0.25, 0.5],
            [0.5, 0.5],
            [0.75, 0.5],
            [1, 0.5],
        ]
    )


def try_promote_to_curved_pipe(
    metrics: SimpleNamespace,
    moving_window: int,
    min_samples_for_check: int = 20,
    min_category_rate: float = 0.90,
) -> bool:
    """
    When every task category has enough recent data and min completion rate is high,
    enable curved pipe and persist the flag. Returns True if promotion occurred.
    """
    task_rates: dict[str, float] = {}
    all_ready = True
    for task_key in TASK_CATEGORY_KEYS:
        history = metrics.task_category_completion_history[task_key]
        if len(history) >= min_samples_for_check:
            recent = history[-min(len(history), moving_window) :]
            task_rates[task_key] = float(np.mean(recent))
        else:
            all_ready = False
            break

    if not all_ready:
        return False

    min_task_rate = min(task_rates.values())
    if min_task_rate < min_category_rate:
        return False

    print(
        f"\n========== All task category completion rates are >= {min_category_rate:.0%}; "
        f"minimum completion rate: {min_task_rate:.2%} =========="
    )
    print("Task completion rate details:")
    for task_key, rate in sorted(task_rates.items()):
        print(f"  {task_key}: {rate:.2%}")
    print("Switching to the curved pipe...")
    metrics.use_curved_pipe = True
    save_use_curved_pipe_flag(metrics.use_curved_pipe_file, metrics.use_curved_pipe)
    print("Switched to the curved pipe")
    print("=" * 70)
    return True
