from data_center_cooling_v0 import SimpleDataCenterEnv, SimpleCoolingCfg

# Create environment with renderer
env = SimpleDataCenterEnv(SimpleCoolingCfg(), render_mode="human")

obs, info = env.reset()
done = False
trunc = False

while not (done or trunc):
    # Random action (cooling power between 0 and 1)
    action = env.action_space.sample()

    # Step environment
    obs, reward, done, trunc, info = env.step(action)

env.close()
