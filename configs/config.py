class Config:
    # ================= Environment =================
    ENV_ID = "parking-v0"

    TOTAL_STEPS = 200_000
    MAX_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 250

    # ================= Common =================
    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 64
    BUFFER_SIZE = 50_000

    # ================= DQN =================
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995
    TARGET_UPDATE = 10

    # ================= TD3 =================
    TAU = 0.005              # Soft update coefficient
    POLICY_NOISE = 0.2       # Target policy smoothing noise
    NOISE_CLIP = 0.5         # Clip noise range
    POLICY_FREQ = 2          # Delayed policy update frequency
    EXPL_NOISE = 0.1         # Exploration noise (action noise)