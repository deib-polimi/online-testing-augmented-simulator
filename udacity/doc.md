# Udacity Gym Environment

## Udacity Gym

Create a Gym:
```python
from udacity.gym import UdacityGym
# Start Simulator
simulator = UdacitySimulator(
    sim_exe_path=simulator_exe_path,
    host=host,
    port=port,
)
simulator.start()

# Create Gym
env = UdacityGym(
    simulator=simulator,
    track="lake",
)
```

Run an experiment:

```python
observation, _ = env.reset(track="lake")

for _ in range(200):
    action = agent(observation)
    observation, reward, terminated, truncated, info = env.step(action)
```
