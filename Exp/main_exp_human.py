"""
Human-in-the-loop manual control experiment script.

Use an interactive UI to control 4 action parameters (flux, frequency, pitch, direction),
reuse the environment logic from `real_world_env.py`, and save videos, obs/action history,
and per-segment results.
"""

import numpy as np
import cv2
import time
import os
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from utils import *
import real_world_env as rw_env
from real_world_env import *
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

env_global = None

# UI state shared by trackbar callbacks and the main loop.
ui_action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
ui_trackbar_window = "Control Sliders"
ui_visual_window = "Action Visualization"
ui_width = 700
ui_height = 650
ui_need_redraw = True


def prepare_human_experiment_folder(output_dir, experiment_config):
    """Create the output folder for a human-controlled run (README, JSON config, etc.)."""
    os.makedirs(output_dir, exist_ok=True)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "# Human-controlled experiment output\n\n"
            "This folder is auto-created with a timestamp each time `main_exp_human.py` starts.\n\n"
            "- `experiment_config.json`: per-run sequence configuration (no model)\n"
            "- `episode_*`: recorded videos (origin / frame / frame_no_text / without) and corresponding `*_info.txt`\n"
            "- `all_episodes_actions.xlsx`: action log for the full run (generated on normal exit)\n"
            "- `obs_action_history.npz`: step-wise observations, human actions, segment indices, and timestamps (incrementally saved after each segment)\n"
            "- `segment_results.xlsx`: per-segment targets, timeout/success, final state, errors, etc. (incrementally saved after each segment)\n"
        )
    cfg_path = os.path.join(output_dir, "experiment_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)


def save_incremental_results(
    output_dir, hist_obs, hist_action, hist_segment, hist_time, segment_result_rows
):
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
    """Handle interrupt signals and stop the coil."""
    print("\nInterrupt received. Stopping coil...")
    global env_global
    if env_global is not None and hasattr(env_global, "coil"):
        try:
            env_global.coil.stop()
        except:
            pass
    cv2.destroyAllWindows()
    sys.exit(0)


def on_trackbar_flux(val):
    """Flux trackbar callback (0-100 -> 0.0-1.0)."""
    global ui_action, ui_need_redraw
    ui_action[0] = val / 100.0
    ui_need_redraw = True


def on_trackbar_freq(val):
    """Frequency trackbar callback (0-100 -> 0.0-1.0)."""
    global ui_action, ui_need_redraw
    ui_action[1] = val / 100.0
    ui_need_redraw = True


def on_trackbar_pitch(val):
    """Pitch trackbar callback (0-180 -> 0.0-1.0 over a half circle)."""
    global ui_action, ui_need_redraw
    ui_action[2] = val / 180.0
    ui_need_redraw = True


def on_trackbar_direction(val):
    """Direction trackbar callback (0-360 -> 0.0-1.0 over a full circle)."""
    global ui_action, ui_need_redraw
    ui_action[3] = val / 360.0
    ui_need_redraw = True


def create_control_ui():
    """Create the human-control UI (trackbars + visualization window)."""
    cv2.namedWindow(ui_trackbar_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ui_trackbar_window, 600, 200)
    
    dummy_img = np.ones((200, 600, 3), dtype=np.uint8) * 240
    cv2.putText(
        dummy_img,
        "Use sliders above to control action parameters",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 100, 100),
        2,
    )
    cv2.imshow(ui_trackbar_window, dummy_img)

    cv2.createTrackbar("Flux (0-100)", ui_trackbar_window, 50, 100, on_trackbar_flux)
    cv2.createTrackbar("Freq (0-100)", ui_trackbar_window, 50, 100, on_trackbar_freq)
    cv2.createTrackbar("Pitch (0-180)", ui_trackbar_window, 90, 180, on_trackbar_pitch)
    cv2.createTrackbar(
        "Direction (0-360)", ui_trackbar_window, 180, 360, on_trackbar_direction
    )

    cv2.namedWindow(ui_visual_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ui_visual_window, ui_width, ui_height)

    ui_action[0] = 0.5
    ui_action[1] = 0.5
    ui_action[2] = 0.5
    ui_action[3] = 0.5


def draw_control_panel():
    """Render the current action parameters (bars + angle indicators)."""
    global ui_need_redraw
    if not ui_need_redraw:
        return
    ui_need_redraw = False

    panel = np.ones((ui_height, ui_width, 3), dtype=np.uint8) * 250

    flux_val = ui_action[0]
    freq_val = ui_action[1]
    pitch_val = ui_action[2]
    direction_val = ui_action[3]

    y_offset = 40
    bar_x_start = 200
    bar_width = 400
    bar_height = 30

    cv2.putText(
        panel,
        f"Flux: {flux_val:.3f}",
        (20, y_offset + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.rectangle(
        panel,
        (bar_x_start, y_offset),
        (bar_x_start + bar_width, y_offset + bar_height),
        (180, 180, 180),
        2,
    )
    filled_width = int(flux_val * bar_width)
    if filled_width > 0:
        cv2.rectangle(
            panel,
            (bar_x_start, y_offset),
            (bar_x_start + filled_width, y_offset + bar_height),
            (100, 200, 100),
            -1,
        )
    cv2.putText(
        panel,
        f"{flux_val*20:.2f} mT",
        (bar_x_start + bar_width + 10, y_offset + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )

    y_offset += 80
    cv2.putText(
        panel,
        f"Freq: {freq_val:.3f}",
        (20, y_offset + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    cv2.rectangle(
        panel,
        (bar_x_start, y_offset),
        (bar_x_start + bar_width, y_offset + bar_height),
        (180, 180, 180),
        2,
    )
    filled_width_freq = int(freq_val * bar_width)
    if filled_width_freq > 0:
        cv2.rectangle(
            panel,
            (bar_x_start, y_offset),
            (bar_x_start + filled_width_freq, y_offset + bar_height),
            (100, 150, 255),
            -1,
        )
    cv2.putText(
        panel,
        f"{freq_val*40:.1f} Hz",
        (bar_x_start + bar_width + 10, y_offset + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )

    y_offset += 100
    pitch_deg = pitch_val * 180.0
    cv2.putText(
        panel,
        f"Pitch: {pitch_val:.3f} ({pitch_deg:.1f} deg)",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    center_pitch = (350, y_offset + 80)
    radius_pitch = 60
    cv2.ellipse(
        panel,
        center_pitch,
        (radius_pitch, radius_pitch),
        0,
        180,
        0,
        (150, 150, 150),
        3,
    )
    angle_pitch_rad = np.deg2rad(180 - pitch_deg)
    end_x = int(center_pitch[0] + radius_pitch * np.cos(angle_pitch_rad))
    end_y = int(center_pitch[1] - radius_pitch * np.sin(angle_pitch_rad))
    cv2.line(panel, center_pitch, (end_x, end_y), (0, 0, 255), 4)
    cv2.circle(panel, (end_x, end_y), 8, (0, 0, 200), -1)

    y_offset += 200
    direction_deg = direction_val * 360.0
    cv2.putText(
        panel,
        f"Direction: {direction_val:.3f} ({direction_deg:.1f} deg)",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    center_dir = (350, y_offset + 80)
    radius_dir = 60
    cv2.circle(panel, center_dir, radius_dir, (150, 150, 150), 3)
    angle_dir_rad = np.deg2rad(90 - direction_deg)
    end_x_dir = int(center_dir[0] + radius_dir * np.cos(angle_dir_rad))
    end_y_dir = int(center_dir[1] - radius_dir * np.sin(angle_dir_rad))
    cv2.line(panel, center_dir, (end_x_dir, end_y_dir), (255, 0, 0), 4)
    cv2.circle(panel, (end_x_dir, end_y_dir), 8, (200, 0, 0), -1)

    cv2.imshow(ui_visual_window, panel)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    env = realEnv()
    env_global = env

    print("Observation space shape:", env.observation_space.shape)
    print("\n" + "="*60)
    print("Human control mode: use UI sliders to control 4 action parameters")
    print("="*60)
    print("\nInstructions:")
    print("1. 'Control Sliders': adjust Flux/Freq/Pitch/Direction with sliders")
    print("2. 'Action Visualization': live visualization of the current action")
    print("3. 'Result Image': environment visualization (robot position/trajectory, etc.)")
    print("4. Press 'q' or ESC to exit the current segment")
    print("="*60 + "\n")

    sequential_target_location_milli = [random.uniform(0.1, 0.9) for _ in range(12)]
    sequential_target_location_nano = [random.uniform(0.1, 0.9) for _ in range(12)]
    sequential_target_release_amount = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sequential_path_milli = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sequential_path_nano = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _project_root = Path(__file__).resolve().parent
    output_dir = str(_project_root / "output" / f"run_human_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    experiment_config = {
        "run_id": run_id,
        "control_mode": "human",
        "sequential_target_location_milli": sequential_target_location_milli,
        "sequential_target_location_nano": sequential_target_location_nano,
        "sequential_target_release_amount": sequential_target_release_amount,
        "sequential_path_milli": sequential_path_milli,
        "sequential_path_nano": sequential_path_nano,
    }
    prepare_human_experiment_folder(output_dir, experiment_config)
    env.video_output_dir = output_dir
    rw_env.video_output_dir = output_dir
    print(f"Output directory (videos/config/data): {os.path.abspath(output_dir)}")

    SEGMENT_TIME_LIMIT_S = 30.0

    create_control_ui()

    try:
        env.activate_devices()
        observation = env.reset(
            episode_number=1,
            target_location_milli=0,
            target_location_nano=0,
            target_release_amount=0,
            plot_nano=True,
        )

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

        action_data = pd.DataFrame(
            columns=["Time", "Action_0", "Action_1", "Action_2", "Action_3"]
        )

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

            print(
                f"\n========== Segment {i} ==========\n"
                f"Target milli: {target_location_milli:.2f}, nano: {target_location_nano:.2f}, release: {target_release_amount:.2f}\n"
                f"Adjust the action with the control panel. Press any key to continue (or wait for timeout).\n"
            )

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

                draw_control_panel()
                key = cv2.waitKey(10)
                if key == ord('q') or key == 27:
                    print("Exit requested by user.")
                    break

                action = ui_action.copy()

                now_time = time.time() - init_time
                times.append(now_time)

                theta_milli = observation[6]
                theta_nano = observation[7]
                action_milli, action_nano = calculate_action(
                    action.copy(), theta_milli, theta_nano
                )
                # flux(mT), freq(Hz), pitch(deg), direction(deg)
                scale = np.array([20, 40, 180, 360])
                print(
                    f"[Step {step_number}] normalized action: {np.round(action, 4)}"
                )
                print(
                    f"        global action (physical): flux={action[0]*20:.2f}mT, freq={action[1]*40:.1f}Hz, pitch={action[2]*180:.1f}°, direct={action[3]*360:.1f}°"
                )
                print(
                    f"        milli action (physical): {np.round(action_milli * scale, 2)}"
                )
                print(
                    f"        nano action  (physical): {np.round(action_nano * scale, 2)}"
                )

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

            cv2.destroyWindow("Result Image")

            print(
                "segment ",
                i,
                "score %.1f" % score,
                "elapsed: %.1f s" % segment_elapsed_s,
                "success"
                if completed_success
                else ("timeout" if segment_timed_out else "end"),
            )

            save_incremental_results(
                output_dir, hist_obs, hist_action, hist_segment, hist_time, segment_result_rows
            )

        finalize_video()

        action_data = pd.DataFrame(
            {
                "Time": times,
                "Action_0": [action[0] for action in actions],
                "Action_1": [action[1] for action in actions],
                "Action_2": [action[2] for action in actions],
                "Action_3": [action[3] for action in actions],
            }
        )

        print("times", times)
        print("actions", actions)

        action_data.to_excel(
            os.path.join(output_dir, "all_episodes_actions.xlsx"), index=False
        )

        print(f"\nExperiment completed. All data saved to: {output_dir}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving completed segment data...")
        save_incremental_results(
            output_dir, hist_obs, hist_action, hist_segment, hist_time, segment_result_rows
        )
        if hasattr(env, "coil"):
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
        if hasattr(env, "coil"):
            env.coil.stop()
    finally:
        if hasattr(env, "coil"):
            try:
                env.coil.stop()
            except:
                pass
        cv2.destroyAllWindows()
