import numpy as np
from .skill import Skill

# Skill 0
def skill_0(length=20):
    return Skill(
        [np.array([-0.00490076, 0.01029501], dtype=np.float32) for _ in range(length)],
        name="skill_0"
    )

# Skill 1
def skill_1(length=20):
    return Skill(
        [np.array([0.03402245, -0.0139027], dtype=np.float32) for _ in range(length)],
        name="skill_1"
    )

# Skill 2
def skill_2(length=20):
    return Skill(
        [np.array([-0.00837033, 0.00049341], dtype=np.float32) for _ in range(length)],
        name="skill_2"
    )

# Skill 3
def skill_3(length=20):
    return Skill(
        [np.array([-0.01968046, -0.00393594], dtype=np.float32) for _ in range(length)],
        name="skill_3"
    )

# Skill 4
def skill_4(length=20):
    return Skill(
        [np.array([0.01094938, 0.01287344], dtype=np.float32) for _ in range(length)],
        name="skill_4"
    )