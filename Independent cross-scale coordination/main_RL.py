from utils.rl_agent import Agent
import numpy as np
import cv2
import time
import random
import os
from sim_env import simEnv
from utils.training_storage import (
    TASK_CATEGORY_KEYS,
    load_training_histories,
    refresh_score_plot_if_due,
    refresh_task_plots_if_due,
    save_best_score,
    save_episode_metrics,
)
from utils.task_settings import (
    DynamicSamplingConfig,
    apply_info_to_episode_flags,
    category_task_completed,
    compute_dynamic_task_distributions,
    infer_task_category,
    print_dynamic_sampling_update,
    sample_pipe_control_points,
    try_promote_to_curved_pipe,
)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MOVING_WINDOW = 50
PLOT_UPDATE_INTERVAL = 100
MODEL_SAVE_INTERVAL = 1000


if __name__ == '__main__':
    tmp_dir = os.path.join('tmp', 'agent')
    video_dir = os.path.join('tmp', 'video')

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    env = simEnv()
    fix_distance = False

    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    print(env.observation_space.shape, env.action_space.shape[0])
    n_games = 600000

    metrics = load_training_histories(tmp_dir, env.reward_range[0])

    thresholds = env.get_release_thresholds()
    print(
        f"Release thresholds: difference <= {thresholds[0]} when target==0, "
        f"difference <= {thresholds[1]} when target==1"
    )

    if os.path.isfile(metrics.use_curved_pipe_file):
        print(f'Loaded pipe type: {"curved pipe" if metrics.use_curved_pipe else "straight pipe"}')
    else:
        print('Initialized pipe type: straight pipe')

    load_checkpoint = False

    if os.path.isfile(os.path.join(tmp_dir, 'current_episode.npy')):
        print(f'Resuming from episode {metrics.start_episode}')

    actor_model_path = os.path.join(tmp_dir, 'actor_last')
    if os.path.exists(actor_model_path):
        agent.load_models_last()
        print('Loaded the last saved model')
    else:
        print('No saved model found; training will start from scratch')

    task_distribution = {'both_exists': 0.8, 'millicore_only': 0.1, 'nano_only': 0.1}
    release_distribution = {0: 0.5, 1: 0.5}
    sampling_cfg = DynamicSamplingConfig()

    for i in range(metrics.start_episode, n_games):
        start_time = time.time()

        target_location_millicore = random.uniform(0, 1)
        target_location_nano = random.uniform(0, 1)

        if i >= sampling_cfg.min_task_samples:
            has_enough_data = all(
                len(metrics.task_category_completion_history[k]) >= sampling_cfg.min_task_samples
                for k in TASK_CATEGORY_KEYS
            )
            if has_enough_data:
                task_distribution, group_rates, release_distribution, release_rates = (
                    compute_dynamic_task_distributions(
                        metrics.task_category_completion_history,
                        task_distribution,
                        release_distribution,
                        sampling_cfg,
                    )
                )
                if i % 100 == 0:
                    print_dynamic_sampling_update(
                        i, task_distribution, group_rates, release_distribution, release_rates
                    )

        exist_index = random.uniform(0, 1)
        if exist_index < task_distribution['both_exists']:
            millicore_exists = True
            nano_exists = True
        elif exist_index < task_distribution['both_exists'] + task_distribution['millicore_only']:
            millicore_exists = True
            nano_exists = False
            target_location_nano = -1
        else:
            millicore_exists = False
            nano_exists = True
            target_location_millicore = -1

        release_rand = random.uniform(0, 1)
        target_release_amount = 0 if release_rand < release_distribution[0] else 1

        pipe_points = sample_pipe_control_points(metrics.use_curved_pipe)

        observation = env.reset(
            target_location_millicore=target_location_millicore,
            target_location_nano=target_location_nano,
            target_release_amount=target_release_amount,
            control_points=pipe_points,
            millicore_exists=millicore_exists,
            nano_exists=nano_exists,
            fix_distance=fix_distance,
        )
        done = False
        score = 0

        if i % MODEL_SAVE_INTERVAL == 0:
            render_on = True
            agent.save_models_last()
        else:
            render_on = False

        episode_millicore_completed = False
        episode_nano_completed = False
        episode_release_completed = False
        episode_all_tasks_completed = False
        episode_millicore_error = None
        episode_nano_error = None
        episode_release_error = None
        last_info = {}

        while not done:
            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            score += reward

            if 'millicore_reached' in info:
                last_info = info
                (
                    episode_millicore_completed,
                    episode_nano_completed,
                    episode_release_completed,
                    episode_all_tasks_completed,
                    episode_millicore_error,
                    episode_nano_error,
                    episode_release_error,
                ) = apply_info_to_episode_flags(info)

            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()

            observation = observation_
            if render_on:
                env.render(mode='human', save_video=True, video_name=f'{len(metrics.mean_score_history)}.mp4')

        if not last_info and 'millicore_reached' in info:
            (
                episode_millicore_completed,
                episode_nano_completed,
                episode_release_completed,
                episode_all_tasks_completed,
                episode_millicore_error,
                episode_nano_error,
                episode_release_error,
            ) = apply_info_to_episode_flags(info)

        cv2.destroyAllWindows()

        if render_on:
            env.close_video()

        score = float(np.asarray(score).item())
        metrics.score_history.append(score)
        avg_score = float(np.mean([float(s) for s in metrics.score_history[-MOVING_WINDOW:]]))
        metrics.mean_score_history.append(avg_score)

        metrics.millicore_completion_history.append(1 if episode_millicore_completed else 0)
        metrics.nano_completion_history.append(1 if episode_nano_completed else 0)
        metrics.release_completion_history.append(1 if episode_release_completed else 0)
        metrics.all_tasks_completion_history.append(1 if episode_all_tasks_completed else 0)

        metrics.millicore_error_history.append(episode_millicore_error)
        metrics.nano_error_history.append(episode_nano_error)
        metrics.release_error_history.append(episode_release_error)

        if metrics.use_curved_pipe:
            metrics.curved_pipe_completion_history.append(1 if episode_all_tasks_completed else 0)
        else:
            metrics.straight_pipe_completion_history.append(1 if episode_all_tasks_completed else 0)

        task_category = infer_task_category(millicore_exists, nano_exists, target_release_amount)
        if task_category:
            task_completed = category_task_completed(
                task_category,
                episode_millicore_completed,
                episode_nano_completed,
                episode_release_completed,
            )
            metrics.task_category_completion_history[task_category].append(1 if task_completed else 0)
            metrics.task_category_episode_numbers[task_category].append(i)

        if i >= 100 and avg_score > metrics.best_score:
            metrics.best_score = avg_score
            save_best_score(tmp_dir, metrics.best_score)
            render_on = True
            if not load_checkpoint:
                agent.save_models_best()

        refresh_score_plot_if_due(metrics, tmp_dir, i, PLOT_UPDATE_INTERVAL, MOVING_WINDOW)
        save_episode_metrics(tmp_dir, i + 1, metrics)

        if i >= MOVING_WINDOW and len(metrics.millicore_completion_history) >= MOVING_WINDOW:
            millicore_success_rate = float(np.mean(metrics.millicore_completion_history[-MOVING_WINDOW:]))
            nano_success_rate = float(np.mean(metrics.nano_completion_history[-MOVING_WINDOW:]))
            release_success_rate = float(np.mean(metrics.release_completion_history[-MOVING_WINDOW:]))
            all_tasks_success_rate = float(np.mean(metrics.all_tasks_completion_history[-MOVING_WINDOW:]))

            refresh_task_plots_if_due(metrics, tmp_dir, i, PLOT_UPDATE_INTERVAL, MOVING_WINDOW)

            if not metrics.use_curved_pipe and i >= MOVING_WINDOW:
                try_promote_to_curved_pipe(metrics, MOVING_WINDOW)

            pipe_type_success_rate = 0.0
            if metrics.use_curved_pipe and len(metrics.curved_pipe_completion_history) >= MOVING_WINDOW:
                pipe_type_success_rate = float(np.mean(metrics.curved_pipe_completion_history[-MOVING_WINDOW:]))
            elif not metrics.use_curved_pipe and len(metrics.straight_pipe_completion_history) >= MOVING_WINDOW:
                pipe_type_success_rate = float(np.mean(metrics.straight_pipe_completion_history[-MOVING_WINDOW:]))

            pipe_type_str = "curved pipe" if metrics.use_curved_pipe else "straight pipe"
            print(f'episode {i}, score {float(score):.1f}, avg_score {float(avg_score):.1f}, '
                  f'Millicore rate: {millicore_success_rate:.2%}, Nanounits rate: {nano_success_rate:.2%}, '
                  f'Release Rate: {release_success_rate:.2%}, All Tasks Rate: {all_tasks_success_rate:.2%}, '
                  f'Pipe type: {pipe_type_str}, {pipe_type_str} completion rate: {pipe_type_success_rate:.2%}, '
                  f'Release thresholds: (0:{thresholds[0]}, 1:{thresholds[1]}), '
                  f'time: {int(time.time()-start_time)}s, target_release_amount: {target_release_amount}, '
                  f'millicore_exists: {millicore_exists}, nano_exists: {nano_exists}')
        else:
            print(
                'episode ', i, 'score %.1f' % float(score), 'avg_score %.1f' % float(avg_score),
                'time:', int(time.time() - start_time), 's', 'target_release_amount:', target_release_amount,
                'millicore_exists:', millicore_exists, 'nanounits_exists:', nano_exists,
            )
            