# Data Center Cooling Optimization using Reinforcement Learning

A simulated data-center cooling control system using Gymnasium and Soft Actor-Critic (SAC) reinforcement learning.

## Project Structure

- data_center_cooling_v0.py     → custom Gymnasium environment  
- train_sac.py                  → SAC training script  
- eval_and_render.py            → evaluation & visualization  
- test_env_run.py               → run the environment without RL (sanity test)

## 1. Test the Environment

To test env: -> python test_env_run

- Temperature
- Cooling Action
- Power usage
<img width="1920" height="1023" alt="image" src="https://github.com/user-attachments/assets/6f44c023-7efa-4a97-99ca-58c791b0c1ce" />


## 2. Core Logic
The environment simulates room temperature using a simple heat balance equation:
<img width="872" height="51" alt="image" src="https://github.com/user-attachments/assets/33a59e32-4756-4c9e-a1e3-0caae025ed81" />
Where:

- HeatIT increases with IT load (0–1)

- Coolingaction increases when RL agent raises cooling effort (0–1)

- Leakambient is a small drift toward outside temperature

- k is a scaling factor controlling system responsiveness

## 3. How to Train (SAC RL)
Train the RL agent using -> python train_sac.py

- Creates the environment
- Normalizes observations
- Trains a Soft Actor-Critic agent

Saves:
sac_dc.zip (policy)
vecnorm.pkl (normalization stats)

## 4. How to Evaluate (Render the Learned Policy)

Run -> python eval_and_render.py

This will:

- Load the trained SAC model
- Run one full episode
- Render 3 live plots: Temperature, Cooling effort, Power usage
- Save eval_plot.png

<img width="1920" height="1019" alt="image" src="https://github.com/user-attachments/assets/c83f089a-b9c8-4bcc-a68d-16db22a0bddd" />

