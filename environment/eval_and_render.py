import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import environment class
from data_center_cooling_v0 import SimpleDataCenterEnv, SimpleCoolingCfg

MODEL_PATH = "sac_dc"        # model save name 
VECNORM_PATH = "vecnorm.pkl" # optional normalization file
EPISODE_MAX_STEPS = None     # None -> uses env horizon or set an int to override

def make_env():
    # Create environment configured to render
    return SimpleDataCenterEnv(SimpleCoolingCfg(), render_mode="human")

def step_with_flexible_unpack(venv, action):
    """
    Call venv.step(action) and normalize return to:
      obs, reward, terminated, truncated, info
    Works with both 4-tuple (obs, rew, done, info) and
    5-tuple (obs, rew, terminated, truncated, info).
    Also handles vectorized env outputs (arrays/lists).
    """
    out = venv.step(action)
    # If a tuple-like object already length 5, return as-is
    if isinstance(out, (list, tuple)) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, (list, tuple)) and len(out) == 4:
        # (obs, reward, done, info) -> split done into (terminated=False, truncated=done)
        obs, reward, done, info = out
        # Some wrappers give 'done' as array-like, keep it consistent
        terminated = False
        truncated = done
    else:
        # Unexpected shape
        raise ValueError("venv.step returned unexpected structure: type=%s len=%s" % (type(out), getattr(out, "__len__", lambda: None)()))

    # If vectorized (n_envs>1), unwrap arrays to element 0
    # obs might be numpy array with batch dim
    def unwrap(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            try:
                # convert 0-d/1-d handled gracefully
                return x[0]
            except Exception:
                return x
        return x

    obs_u = unwrap(obs)
    # reward may be array/np.ndarray -> convert to scalar or element 0
    reward_u = unwrap(reward)
    terminated_u = unwrap(terminated)
    truncated_u = unwrap(truncated)
    info_u = unwrap(info)

    return obs_u, reward_u, bool(terminated_u), bool(truncated_u), info_u

def main():
    # 1) Build a base (plain) env so we can access hist buffers for plotting
    plain_venv = DummyVecEnv([make_env])

    # 2) Load VecNormalize if exists (so observations match training)
    try:
        venv = VecNormalize.load(VECNORM_PATH, plain_venv)
        print(f"Loaded VecNormalize from '{VECNORM_PATH}'.")
    except Exception as e:
        venv = plain_venv
        print("VecNormalize not loaded (missing or error). Proceeding without normalization.")
        # print(e)

    # 3) Load trained model (attach env for convenience)
    try:
        model = SAC.load(MODEL_PATH, env=venv)
        print(f"Loaded model from '{MODEL_PATH}'.")
    except Exception as e:
        print("Failed to load model. Make sure sac_dc.zip (or sac_dc.zip) exists in the working directory.")
        raise

    # 4) Reset env
    obs = venv.reset()
    # If vectorized, obs may be array with batch dim; unwrap to single env case
    if isinstance(obs, (list, tuple, np.ndarray)):
        try:
            obs = obs[0]
        except Exception:
            pass

    # fetch underlying base env (first env in plain_venv)
    base_env = plain_venv.envs[0]

    # compute horizon if not overridden
    if EPISODE_MAX_STEPS is not None:
        max_steps = EPISODE_MAX_STEPS
    else:
        steps_per_hour = int(3600 / base_env.cfg.dt_sec)
        max_steps = steps_per_hour * base_env.cfg.hours_per_ep

    print(f"Starting evaluation run for up to {max_steps} steps... (press Ctrl+C to stop)")

    step = 0
    try:
        while step < max_steps:
            # model.predict expects vectorized obs if env is vec; give it as batch if needed
            # venv is vectorized; we must pass obs in the same shape it expects.
            # Easiest approach: pass the latest observation returned by venv (we kept 'obs' from reset)
            # but ensure shape matches vec env: wrap in batch dim
            obs_batch = np.expand_dims(obs, axis=0) if not getattr(obs, "shape", None) or len(np.shape(obs)) == 1 else obs
            action, _ = model.predict(obs_batch, deterministic=True)

            # step and unpack robustly
            obs_new, reward, terminated, truncated, info = step_with_flexible_unpack(venv, action)

            step += 1
            obs = obs_new  # update obs for next predict

            if terminated or truncated:
                print(f"Episode finished at step {step} (terminated={terminated}, truncated={truncated})")
                break

            # tiny sleep so the renderer (plt.pause inside env.render) has time
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")

    # Save snapshot using history stored on base_env (the actual SimpleDataCenterEnv instance)
    hist_t = getattr(base_env, "_hist_t", None)
    hist_T = getattr(base_env, "_hist_T", None)
    hist_a = getattr(base_env, "_hist_a", None)
    hist_P = getattr(base_env, "_hist_P", None)

    if hist_t and hist_T and hist_a and hist_P:
        print("Saving snapshot plot to 'eval_plot.png' ...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax1.plot(hist_t, hist_T, label="T_room (°C)")
        ax1.axhline(base_env.cfg.target_C, linestyle="--", color="gray", label="Target")
        ax1.axhline(base_env.cfg.Tmax_C, linestyle=":", color="red", label="Tmax")
        ax1.set_ylabel("Temperature (°C)")
        ax1.legend()

        ax2.plot(hist_t, hist_a, label="Action (cooling effort)")
        ax2.set_ylabel("Action (0..1)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()

        ax3.plot(hist_t, hist_P, label="Power (kW)")
        ax3.set_ylabel("Power (kW)")
        ax3.set_xlabel("Time (hours)")
        ax3.legend()

        fig.tight_layout()
        fig.savefig("eval_plot.png", dpi=150)
        print("Saved eval_plot.png")
    else:
        print("No history data found in the environment (hist arrays missing).")

    # Close environments cleanly
    try:
        venv.close()
        plain_venv.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
