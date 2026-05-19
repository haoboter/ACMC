"""
Main loop for running the real-world environment with a trained RL agent.
"""

from rl_agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import random
import os
import json
import shutil
from pathlib import Path
import signal
import sys
from datetime import datetime
from utils import *
import real_world_env as rw_env
from real_world_env import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

env_global = None


def prepare_experiment_run_folder(output_dir, agent_checkpoint_dir, experiment_config):
    """
    Create the per-run output folder and snapshot the loaded checkpoints.

    Side effects:
    - creates `output_dir`
    - writes `README.md` and `experiment_config.json`
    - copies `agent_checkpoint_dir` into `loaded_agent_models/` if it exists
    """
    os.makedirs(output_dir, exist_ok=True)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "# Experiment run output\n\n"
            "This folder is auto-created with a timestamp each time `main_exp.py` starts.\n\n"
            "- `experiment_config.json`: run configuration and agent checkpoint load path\n"
            "- `loaded_agent_models/`: snapshot of checkpoints copied at launch\n"
            "- `episode_*`: recorded videos (origin / frame / frame_no_text / without) and corresponding `*_info.txt`\n"
            "- `all_episodes_actions.xlsx`: action log for the full run (generated on normal exit)\n"
            "- `obs_action_history.npz`: step-wise observations (policy inputs), actions, segment indices, and timestamps (incrementally saved after each segment)\n"
            "- `segment_results.xlsx`: per-segment targets, timeout/success, final state, errors, etc. (incrementally saved after each segment)\n"
        )
    cfg_path = os.path.join(output_dir, "experiment_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)
    dest_models = os.path.join(output_dir, "loaded_agent_models")
    abs_ckpt = os.path.abspath(agent_checkpoint_dir)
    if os.path.isdir(abs_ckpt):
        shutil.copytree(abs_ckpt, dest_models, dirs_exist_ok=True)
    else:
        print(f"Warning: agent checkpoint directory not found, skip copy: {abs_ckpt}")


def save_incremental_results(output_dir, hist_obs, hist_action, hist_segment, hist_time, segment_result_rows):
    """Write NPZ/Excel after each segment to avoid losing completed data on interruption."""
    if hist_obs:
        np.savez_compressed(
            os.path.join(output_dir, "obs_action_history.npz"),
            observations=np.stack(hist_obs, axis=0),
            actions=np.stack(hist_action, axis=0),
            segment_index=np.array(hist_segment, dtype=np.int32),
            time_since_run_start_s=np.array(hist_time, dtype=np.float64),
        )
    if segment_result_rows:
        pd.DataFrame(segment_result_rows).to_excel(
            os.path.join(output_dir, "segment_results.xlsx"), index=False
        )


def signal_handler(sig, frame):
    """Handle interrupts and stop the coil before exiting."""
    print("\nInterrupt signal received, stopping coil...")
    global env_global
    if env_global is not None and hasattr(env_global, 'coil'):
        try:
            env_global.coil.stop()
        except:
            pass
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    env = realEnv()
    env_global = env
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0]) 

    print("Observation space shape:", env.observation_space.shape, "Action dimension:", env.action_space.shape[0])
    n_games = 1001

    os.makedirs('agent', exist_ok=True)
    figure_file = 'agent/hb_env.png'

    if os.path.exists('agent/best_score.npy'):
        best_score = np.load('agent/best_score.npy')
    else:
        best_score = env.reward_range[0]

    score_history = []

    mean_score_history = []

    # Convention: checkpoints and auxiliary files are stored under `agent/`.
    agent_checkpoint_dir = os.path.normpath("./agent")
    
    agent.load_models_last(agent_checkpoint_dir)

    sequential_target_location_milli = [random.uniform(0.1, 0.9) for _ in range(12)]
    sequential_target_location_nano = [random.uniform(0.1, 0.9) for _ in range(12)]
    sequential_target_release_amount = [1, 1, 1, 1, 1,    1, 1, 1, 1, 1, 1, 1]
    
    sequential_path_milli = [0, 0, 0, 0, 0,       0, 0, 0, 0, 0, 0, 0]
    sequential_path_nano = [1, 1, 1, 1, 1,       1, 1, 1, 1, 1, 1, 1]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _project_root = Path(__file__).resolve().parent
    output_dir = str(_project_root / "output" / f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    experiment_config = {
        "run_id": run_id,
        "agent_checkpoint_load_path": agent_checkpoint_dir,
        "sequential_target_location_milli": sequential_target_location_milli,
        "sequential_target_location_nano": sequential_target_location_nano,
        "sequential_target_release_amount": sequential_target_release_amount,
        "sequential_path_milli": sequential_path_milli,
        "sequential_path_nano": sequential_path_nano,
    }
    prepare_experiment_run_folder(output_dir, agent_checkpoint_dir, experiment_config)
    env.video_output_dir = output_dir
    rw_env.video_output_dir = output_dir
    print(f"Output directory (videos/config/checkpoint snapshot): {os.path.abspath(output_dir)}")

    SEGMENT_TIME_LIMIT_S = 30.0

    try:
        env.activate_devices()
        observation = env.reset(episode_number=1, target_location_milli=0,
                                target_location_nano=0, target_release_amount=0,
                                plot_nano=True)

        actions = []
        times = []
        init_time = time.time()

        hist_obs = []
        hist_action = []
        hist_segment = []
        hist_time = []
        segment_result_rows = []

        def err_target_minus_actual(tgt, cur):
            if tgt == -1:
                return np.nan
            return tgt - cur

        action_data = pd.DataFrame(columns=['Time', 'Action_0', 'Action_1', 'Action_2', 'Action_3'])

        for i in range(len(sequential_target_location_milli)):
            start_time = time.time()
            segment_wall_t0 = time.time()
            segment_timed_out = False

            target_location_milli = sequential_target_location_milli[i]
            target_location_nano = sequential_target_location_nano[i]
            target_release_amount = sequential_target_release_amount[i]
            choose_path_milli = sequential_path_milli[i]
            choose_path_nano = sequential_path_nano[i]

            plot_nano = True

            task_info = {
                "segment_index": i,
                "run_start_time": run_id,
                "target_location_milli": target_location_milli,
                "target_location_nano": target_location_nano,
                "target_release_amount": target_release_amount,
                "choose_path_milli": choose_path_milli,
                "choose_path_nano": choose_path_nano,
                "sequential_target_location_milli": sequential_target_location_milli,
                "sequential_target_location_nano": sequential_target_location_nano,
                "sequential_target_release_amount": sequential_target_release_amount,
            }
            env.start_new_video_segment(i, task_info)

            done = False
            done_A = False
            score = 0

            env.fresh_time()

            step_number = 0
            no_progress_count = 0
            other_mode_count = 0
            initial_distance = np.abs(observation[0] - observation[3])
            while not done_A:
                if time.time() - segment_wall_t0 >= SEGMENT_TIME_LIMIT_S:
                    segment_timed_out = True
                    print(
                        f"[Segment {i}] Time limit exceeded ({SEGMENT_TIME_LIMIT_S:.0f}s). Aborting this segment."
                    )
                    break

                step_number += 1
                if step_number % 20 == 0:
                    initial_distance = np.abs(observation[0] - observation[3])

                if np.abs(observation[0] - observation[3]) - initial_distance >= -0.05:
                    no_progress_count += 1

                if target_location_milli == -1:
                    # Convention: if the milli robot is absent, we propagate -1 in its related
                    # observation fields (position/current/angle) to keep logs unambiguous.
                    observation[0] = -1
                    observation[3] = -1
                    observation[6] = -1
                
                if target_location_nano == -1:
                    # Convention: same as above for the nano robot.
                    observation[1] = -1
                    observation[4] = -1
                    observation[7] = -1
                
                obs_policy = np.asarray(observation, dtype=np.float64).copy()
                action = agent.choose_action(observation)


                now_time = time.time() - init_time
                times.append(now_time)

                theta_milli = observation[6]
                theta_nano = observation[7]
                action_milli, action_nano = calculate_action(action.copy(), theta_milli, theta_nano)
                # Unit conversion for readable logging: normalized -> physical units.
                # flux(mT), freq(Hz), pitch(deg), direction(deg)
                scale = np.array([20, 40, 180, 360])
                print(f"[Step {step_number}] normalized global action: {np.round(action, 4)}")
                print(f"        global action (physical): flux={action[0]*20:.2f}mT, freq={action[1]*40:.1f}Hz, pitch={action[2]*180:.1f}°, direct={action[3]*360:.1f}°")
                print(f"        milli action (physical): {np.round(action_milli * scale, 2)}")
                print(f"        nano action  (physical): {np.round(action_nano * scale, 2)}")

                env.update_condition(
                    target_location_milli=target_location_milli,
                    target_location_nano=target_location_nano,
                    target_release_amount=target_release_amount,
                    plot_nano=plot_nano,
                    choose_path_milli=choose_path_milli,
                    choose_path_nano=choose_path_nano,
                )
                observation_, reward, done, done_A = env.step(action=action)

                actions.append(action)
                hist_obs.append(obs_policy)
                hist_action.append(np.asarray(action, dtype=np.float64).copy())
                hist_segment.append(i)
                hist_time.append(now_time)

                cv2.imshow("Result Image", env.result_image)
                cv2.waitKey(1)
                score += reward

                observation = observation_

            segment_elapsed_s = time.time() - start_time
            completed_success = bool(done_A) and not segment_timed_out
            final_milli = float(observation[3])
            final_nano = float(observation[4])
            final_release = float(observation[5])
            mean_gray_end = float(getattr(env, "mean_gray", np.nan))

            em = err_target_minus_actual(target_location_milli, final_milli)
            ec = err_target_minus_actual(target_location_nano, final_nano)
            er = err_target_minus_actual(target_release_amount, final_release)

            segment_result_rows.append(
                {
                    "segment_index": i,
                    "target_location_milli": target_location_milli,
                    "target_location_nano": target_location_nano,
                    "target_release_amount": target_release_amount,
                    "choose_path_milli": choose_path_milli,
                    "choose_path_nano": choose_path_nano,
                    "completed_success": completed_success,
                    "timed_out_30s": segment_timed_out,
                    "elapsed_s": segment_elapsed_s,
                    "control_steps_in_segment": step_number,
                    "final_current_location_milli": final_milli,
                    "final_current_location_nano": final_nano,
                    "final_current_release_amount": final_release,
                    "mean_gray_at_segment_end": mean_gray_end,
                    "error_milli_target_minus_actual": em,
                    "error_nano_target_minus_actual": ec,
                    "error_release_target_minus_actual": er,
                    "abs_error_milli": np.abs(em) if not np.isnan(em) else np.nan,
                    "abs_error_nano": np.abs(ec) if not np.isnan(ec) else np.nan,
                    "abs_error_release": np.abs(er) if not np.isnan(er) else np.nan,
                }
            )

            cv2.destroyAllWindows()

            score_history.append(score)
            avg_score = np.mean(score_history[-50:])
            mean_score_history.append(avg_score)

            if i >= 100 and avg_score > best_score:
                best_score = avg_score
                np.save('agent/best_score.npy', np.array(best_score))
                render_on = True
                # if not load_checkpoint:
                #     agent.save_models_best()

            plt.figure(figsize=(10, 6))
            num_scores = len(mean_score_history)
            x = np.linspace(0, num_scores - 1, num_scores)
            plt.plot(x, mean_score_history, marker='o', linestyle='-', color='b', label='Average score')
            plt.title('Average score over 50 episodes')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.xlim(0, max(num_scores + 1, 1))
            arr = np.array(mean_score_history, dtype=float)
            if arr.size > 0:
                y_min, y_max = np.nanmin(arr), np.nanmax(arr)
                if np.isfinite(y_min) and np.isfinite(y_max):
                    plt.ylim(y_min - 10, y_max + 10)
                else:
                    plt.ylim(0, 100)
            else:
                plt.ylim(0, 100)
            plt.grid()
            plt.legend()
            plt.savefig('agent/score_plot.png')
            plt.close()

            np.save('agent/mean_score_history.npy', np.array(mean_score_history))
            print(
                "segment ",
                i,
                "score %.1f" % score,
                "avg_score %.1f" % avg_score,
                "elapsed: %.1f s" % segment_elapsed_s,
                "success" if completed_success else ("timeout" if segment_timed_out else "end"),
            )

            save_incremental_results(
                output_dir, hist_obs, hist_action, hist_segment, hist_time, segment_result_rows
            )

        finalize_video()

        action_data = pd.DataFrame({
            'Time': times,
            'Action_0': [action[0] for action in actions],
            'Action_1': [action[1] for action in actions],
            'Action_2': [action[2] for action in actions],
            'Action_3': [action[3] for action in actions]
        })

        print('times', times)
        print('actions', actions)

        action_data.to_excel(os.path.join(output_dir, "all_episodes_actions.xlsx"), index=False)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving completed segment data...")
        save_incremental_results(
            output_dir, hist_obs, hist_action, hist_segment, hist_time, segment_result_rows
        )
        if hasattr(env, 'coil'):
            env.coil.stop()
        print("Data saved. Devices stopped.")
    except Exception as e:
        import traceback
        print(f"Exception occurred: {type(e).__name__}: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        print("\nSaving completed segment data...")
        save_incremental_results(
            output_dir, hist_obs, hist_action, hist_segment, hist_time, segment_result_rows
        )
        print("Shutting down devices...")
        if hasattr(env, 'coil'):
            env.coil.stop()
    finally:
        if hasattr(env, 'coil'):
            try:
                env.coil.stop()
            except:
                pass
        cv2.destroyAllWindows()