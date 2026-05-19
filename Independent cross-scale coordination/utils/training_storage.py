"""Load/save training metrics and resume state for a run directory (``tmp_dir`` from the training script); plots export beside those files."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

from .util import plot_score_history, plot_task_completion_rates, plot_task_errors

TASK_CATEGORY_KEYS = ("task_a", "task_b", "task_c", "task_e", "task_f")


def _join(tmp_dir: str, *parts: str) -> str:
    return os.path.join(tmp_dir, *parts)


def load_training_histories(tmp_dir: str, default_best_score) -> SimpleNamespace:
    """
    Restore lists and scalars from disk for plotting and resume.
    Returned list fields are mutable and updated in the training loop.
    """
    if os.path.exists(_join(tmp_dir, "best_score.npy")):
        best_score = np.load(_join(tmp_dir, "best_score.npy"))
    else:
        best_score = default_best_score

    mean_score_history: list[float] = []
    if os.path.isfile(_join(tmp_dir, "mean_score_history.npy")):
        mean_score_history = [
            float(s) for s in np.load(_join(tmp_dir, "mean_score_history.npy")).tolist()
        ]

    score_history: list[float] = []
    if os.path.isfile(_join(tmp_dir, "score_history.npy")):
        score_history = [float(s) for s in np.load(_join(tmp_dir, "score_history.npy")).tolist()]

    def _load_int_list(name: str) -> list[int]:
        path = _join(tmp_dir, name)
        if os.path.isfile(path):
            return [int(x) for x in np.load(path).tolist()]
        return []

    millicore_completion_history = _load_int_list("millicore_completion_history.npy")
    nano_completion_history = _load_int_list("nano_completion_history.npy")
    release_completion_history = _load_int_list("release_completion_history.npy")
    all_tasks_completion_history = _load_int_list("all_tasks_completion_history.npy")

    def _load_object_list(name: str) -> list:
        path = _join(tmp_dir, name)
        if os.path.isfile(path):
            return np.load(path, allow_pickle=True).tolist()
        return []

    millicore_error_history = _load_object_list("millicore_error_history.npy")
    nano_error_history = _load_object_list("nano_error_history.npy")
    release_error_history = _load_object_list("release_error_history.npy")

    task_category_completion_history = {k: [] for k in TASK_CATEGORY_KEYS}
    task_category_episode_numbers = {k: [] for k in TASK_CATEGORY_KEYS}
    for task_key in TASK_CATEGORY_KEYS:
        hf = _join(tmp_dir, f"{task_key}_completion_history.npy")
        ef = _join(tmp_dir, f"{task_key}_episode_numbers.npy")
        if os.path.isfile(hf):
            task_category_completion_history[task_key] = [int(x) for x in np.load(hf).tolist()]
        if os.path.isfile(ef):
            task_category_episode_numbers[task_key] = [int(x) for x in np.load(ef).tolist()]

    use_curved_pipe_file = _join(tmp_dir, "use_curved_pipe.npy")
    if os.path.isfile(use_curved_pipe_file):
        use_curved_pipe = bool(np.load(use_curved_pipe_file))
    else:
        use_curved_pipe = False

    straight_pipe_completion_history = _load_int_list("straight_pipe_completion_history.npy")
    curved_pipe_completion_history = _load_int_list("curved_pipe_completion_history.npy")

    start_episode = 0
    if os.path.isfile(_join(tmp_dir, "current_episode.npy")):
        start_episode = int(np.load(_join(tmp_dir, "current_episode.npy")))

    return SimpleNamespace(
        best_score=best_score,
        score_history=score_history,
        mean_score_history=mean_score_history,
        millicore_completion_history=millicore_completion_history,
        nano_completion_history=nano_completion_history,
        release_completion_history=release_completion_history,
        all_tasks_completion_history=all_tasks_completion_history,
        millicore_error_history=millicore_error_history,
        nano_error_history=nano_error_history,
        release_error_history=release_error_history,
        task_category_completion_history=task_category_completion_history,
        task_category_episode_numbers=task_category_episode_numbers,
        use_curved_pipe=use_curved_pipe,
        use_curved_pipe_file=use_curved_pipe_file,
        straight_pipe_completion_history=straight_pipe_completion_history,
        curved_pipe_completion_history=curved_pipe_completion_history,
        start_episode=start_episode,
    )


def save_best_score(tmp_dir: str, best_score) -> None:
    np.save(_join(tmp_dir, "best_score.npy"), np.array(best_score))


def save_use_curved_pipe_flag(use_curved_pipe_file: str, use_curved_pipe: bool) -> None:
    np.save(use_curved_pipe_file, np.array(use_curved_pipe))


def save_episode_metrics(tmp_dir: str, next_episode: int, m: SimpleNamespace) -> None:
    """Persist episode-level arrays; `m` is the object returned by load_training_histories (mutated in-loop)."""
    np.save(_join(tmp_dir, "mean_score_history.npy"), np.array(m.mean_score_history, dtype=np.float64))
    np.save(_join(tmp_dir, "score_history.npy"), np.array(m.score_history, dtype=np.float64))
    np.save(_join(tmp_dir, "current_episode.npy"), np.array(next_episode))

    np.save(_join(tmp_dir, "millicore_completion_history.npy"), np.array(m.millicore_completion_history))
    np.save(_join(tmp_dir, "nano_completion_history.npy"), np.array(m.nano_completion_history))
    np.save(_join(tmp_dir, "release_completion_history.npy"), np.array(m.release_completion_history))
    np.save(_join(tmp_dir, "all_tasks_completion_history.npy"), np.array(m.all_tasks_completion_history))

    np.save(_join(tmp_dir, "millicore_error_history.npy"), np.array(m.millicore_error_history, dtype=object))
    np.save(_join(tmp_dir, "nano_error_history.npy"), np.array(m.nano_error_history, dtype=object))
    np.save(_join(tmp_dir, "release_error_history.npy"), np.array(m.release_error_history, dtype=object))

    for task_key, history in m.task_category_completion_history.items():
        np.save(_join(tmp_dir, f"{task_key}_completion_history.npy"), np.array(history))
    for task_key, episodes in m.task_category_episode_numbers.items():
        np.save(_join(tmp_dir, f"{task_key}_episode_numbers.npy"), np.array(episodes))

    save_use_curved_pipe_flag(m.use_curved_pipe_file, m.use_curved_pipe)
    np.save(_join(tmp_dir, "straight_pipe_completion_history.npy"), np.array(m.straight_pipe_completion_history))
    np.save(_join(tmp_dir, "curved_pipe_completion_history.npy"), np.array(m.curved_pipe_completion_history))


def refresh_score_plot_if_due(
    metrics: SimpleNamespace,
    tmp_dir: str,
    episode_i: int,
    plot_interval: int,
    moving_window: int,
) -> None:
    if episode_i % plot_interval == 0:
        plot_score_history(
            metrics.mean_score_history,
            os.path.join(tmp_dir, "score_plot.png"),
            window_size=moving_window,
        )


def refresh_task_plots_if_due(
    metrics: SimpleNamespace,
    tmp_dir: str,
    episode_i: int,
    plot_interval: int,
    moving_window: int,
) -> None:
    if episode_i % plot_interval != 0:
        return
    if episode_i < moving_window or len(metrics.millicore_completion_history) < moving_window:
        return
    plot_task_completion_rates(
        metrics.millicore_completion_history,
        metrics.nano_completion_history,
        metrics.release_completion_history,
        metrics.all_tasks_completion_history,
        os.path.join(tmp_dir, "task_completion_rates.png"),
        task_category_completion_history=metrics.task_category_completion_history,
        task_category_episode_numbers=metrics.task_category_episode_numbers,
        current_episode=episode_i,
    )
    if len(metrics.millicore_error_history) >= moving_window:
        plot_task_errors(
            metrics.millicore_error_history,
            metrics.nano_error_history,
            metrics.release_error_history,
            os.path.join(tmp_dir, "task_errors.png"),
            window_size=moving_window,
        )
