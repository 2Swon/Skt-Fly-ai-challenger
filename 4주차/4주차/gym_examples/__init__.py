from gym.envs.registration import register

register(
    id="gym_examples/ShootingAirplane-v0",
    entry_point="gym_examples.envs:ShootingAirplaneEnv"
)

register(
    id="gym_examples/OmokSinglePlayer-v0",
    entry_point="gym_examples.envs:OmokSinglePlayerEnv"
)

