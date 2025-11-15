import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class SimpleCoolingCfg:
    dt_sec: float = 60.0
    target_C: float = 24.0
    Tmax_C: float = 30.0
    ambient_C: float = 28.0
    heat_max_kW: float = 15.0
    cool_max_kW: float = 20.0
    leak_coeff: float = 0.05
    k_dynamics: float = 0.02
    w_energy: float = 0.02
    viol_penalty: float = 10.0
    hours_per_ep: int = 24
    obs_noise: float = 0.0
    randomize_reset: bool = True

class SimpleDataCenterEnv(gym.Env):
    """
    simulation-only RL env for data-center cooling
    - obs: [T_room (°C), IT_load (0..1)]
    - action: cooling effort a in [0,1]
    - reward: track target temperature, minimize energy; penalize overheating
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: SimpleCoolingCfg = None, seed: int = None, render_mode: str = None):
        super().__init__()
        self.cfg = cfg or SimpleCoolingCfg()
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=np.array([0.0], dtype=np.float32),
                                       high=np.array([1.0], dtype=np.float32),
                                       dtype=np.float32)

        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array([60.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Internal state
        self.t = 0
        self.T_room = 24.0
        self.IT_load = 0.5

        # history buffers for plotting
        self._hist_t = []
        self._hist_T = []
        self._hist_a = []
        self._hist_P = []

        #matplotlib figure/axes/line handles
        self._fig = None
        self._ax_temp = None
        self._ax_act = None
        self._ax_pow = None
        self._line_T = None
        self._line_a = None
        self._line_P = None
        self._vline_target = None
        self._vline_tmax = None

    # helpers
    def _obs(self):
        obs = np.array([self.T_room, self.IT_load], dtype=np.float32)
        if self.cfg.obs_noise > 0:
            obs += self.rng.normal(0, self.cfg.obs_noise, size=obs.shape).astype(np.float32)
        return obs

    def _heat_kW(self):
        return self.cfg.heat_max_kW * self.IT_load

    def _cool_kW(self, a):
        return self.cfg.cool_max_kW * a

    def _leak_term(self):
        return self.cfg.leak_coeff * (self.cfg.ambient_C - self.T_room)

    def _step_profiles(self):
        self.IT_load = float(np.clip(self.IT_load + 0.02*self.rng.normal(), 0.1, 0.95))

    # Gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        if self.cfg.randomize_reset:
            self.T_room = float(self.rng.uniform(22.0, 26.0))
            self.IT_load = float(self.rng.uniform(0.3, 0.8))
        else:
            self.T_room, self.IT_load = 24.0, 0.5

        #clear histories when starting a new episode
        self._hist_t.clear()
        self._hist_T.clear()
        self._hist_a.clear()
        self._hist_P.clear()

        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(action[0], 0.0, 1.0))

        heat_in = self._heat_kW()
        cool_out = self._cool_kW(a)
        leak = self._leak_term()
        dT = self.cfg.k_dynamics * (heat_in - cool_out) + leak
        self.T_room += dT

        self._step_profiles()
        self.t += 1

        temp_err = self.T_room - self.cfg.target_C
        power = a * self.cfg.cool_max_kW
        reward = - (temp_err**2 + self.cfg.w_energy * power)
        violation = int(self.T_room > self.cfg.Tmax_C)
        if violation:
            reward -= self.cfg.viol_penalty

        steps_per_hour = int(3600 / self.cfg.dt_sec)
        horizon = steps_per_hour * self.cfg.hours_per_ep
        terminated = False
        truncated = (self.t >= horizon)

        #push to history buffers (time in hours for x-axis)
        t_hours = self.t * (self.cfg.dt_sec / 3600.0)
        self._hist_t.append(t_hours)
        self._hist_T.append(float(self.T_room))
        self._hist_a.append(float(a))
        self._hist_P.append(float(power))

        info = {
            "power_kW": float(power),
            "violation": violation,
            "temp_error": float(temp_err),
            "T_room": float(self.T_room),
            "IT_load": float(self.IT_load),
            "ambient_C": float(self.cfg.ambient_C),
            "action": float(a),
        }

        # live render during training/testing if asked
        if self.render_mode == "human":
            self.render()

        return self._obs(), float(reward), terminated, truncated, info

    #renderer
    def render(self):
        if self.render_mode != "human":
            return

        # Lazily create figure/axes/lines once
        if self._fig is None:
            self._fig = plt.figure(figsize=(9, 8))
            self._ax_temp = self._fig.add_subplot(3, 1, 1)
            self._ax_act  = self._fig.add_subplot(3, 1, 2)
            self._ax_pow  = self._fig.add_subplot(3, 1, 3)

            # Temperature plot
            (self._line_T,) = self._ax_temp.plot([], [], label="T_room (°C)")
            self._ax_temp.axhline(self.cfg.target_C, linestyle="--", label="Target", alpha=0.7)
            self._ax_temp.axhline(self.cfg.Tmax_C, linestyle=":", label="Tmax", alpha=0.7)
            self._ax_temp.set_ylabel("Temperature (°C)")
            self._ax_temp.set_title("Room Temperature")
            self._ax_temp.legend(loc="upper right")

            # Action plot
            (self._line_a,) = self._ax_act.plot([], [], label="Action (cooling effort 0..1)")
            self._ax_act.set_ylim(-0.05, 1.05)
            self._ax_act.set_ylabel("Action")
            self._ax_act.set_title("Cooling Effort")
            self._ax_act.legend(loc="upper right")

            # Power plot
            (self._line_P,) = self._ax_pow.plot([], [], label="Power (kW)")
            self._ax_pow.set_ylabel("kW")
            self._ax_pow.set_xlabel("Time (hours)")
            self._ax_pow.set_title("Cooling Power")
            self._ax_pow.legend(loc="upper right")

            self._fig.tight_layout()

        # Update data
        x = self._hist_t
        self._line_T.set_data(x, self._hist_T)
        self._line_a.set_data(x, self._hist_a)
        self._line_P.set_data(x, self._hist_P)

        # Keep x-limits growing with time autoscale y from data
        for ax in (self._ax_temp, self._ax_act, self._ax_pow):
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(left=max(0.0, (x[-1] - 4.0)) if x else 0.0,  # show last ~4 hours window
                        right=(x[-1] if x else 4.0))

        plt.pause(0.001)  # non-blocking GUI update

    #clean up figure
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax_temp = self._ax_act = self._ax_pow = None
            self._line_T = self._line_a = self._line_P = None
