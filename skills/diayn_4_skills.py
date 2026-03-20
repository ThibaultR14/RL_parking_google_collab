import numpy as np
from .skill import Skill

# Skill 0
def skill_0(length=20):
    return Skill(
        [np.array([0.0197, 0.0221], dtype=np.float32) for _ in range(length)],
        name="skill_0"
    )

# Skill 1
def skill_1(length=20):
    return Skill(
        [np.array([-0.0045, 0.0082], dtype=np.float32) for _ in range(length)],
        name="skill_1"
    )

# Skill 2
def skill_2(length=20):
    return Skill(
        [np.array([0.0246, 0.0319], dtype=np.float32) for _ in range(length)],
        name="skill_2"
    )

# Skill 3
def skill_3(length=20):
    return Skill(
        [np.array([0.0188, 0.0336], dtype=np.float32) for _ in range(length)],
        name="skill_3"
    )