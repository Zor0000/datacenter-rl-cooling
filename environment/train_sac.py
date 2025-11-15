from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data_center_cooling_v0 import SimpleDataCenterEnv, SimpleCoolingCfg

def make_env():
    return SimpleDataCenterEnv(SimpleCoolingCfg())

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    tau=0.02,
    batch_size=256,
    verbose=1
)

model.learn(total_timesteps=300_000)
model.save("sac_dc")
env.save("vecnorm.pkl")
print("Training complete!")
