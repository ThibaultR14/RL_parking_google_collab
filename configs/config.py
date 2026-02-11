class Config:
    ENV_ID = "parking-v0"

    TOTAL_STEPS = 200_000
    MAX_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 250

    GAMMA = 0.99
    LR = 1e-3

    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995

    BATCH_SIZE = 64
    BUFFER_SIZE = 50_000
    TARGET_UPDATE = 10
