import numpy as np
from .skill import Skill

# Skill 0
def skill_0(length=20):
    return Skill(
        [np.array([-0.0500349, -0.02142239], dtype=np.float32) for _ in range(length)],
        name="skill_0"
    )

# Skill 1
def skill_1(length=20):
    return Skill(
        [np.array([0.04860725, -0.05084863], dtype=np.float32) for _ in range(length)],
        name="skill_1"
    )

# Skill 2
def skill_2(length=20):
    return Skill(
        [np.array([-0.0284638, 0.03540613], dtype=np.float32) for _ in range(length)],
        name="skill_2"
    )

# Skill 3
def skill_3(length=20):
    return Skill(
        [np.array([0.03835916, 0.07752454], dtype=np.float32) for _ in range(length)],
        name="skill_3"
    )

# Skill 4
def skill_4(length=20):
    return Skill(
        [np.array([0.00397404, -0.00885314], dtype=np.float32) for _ in range(length)],
        name="skill_4"
    )

# Skill 5
def skill_5(length=20):
    return Skill(
        [np.array([0.04191199, 0.02483143], dtype=np.float32) for _ in range(length)],
        name="skill_5"
    )

# Skill 6
def skill_6(length=20):
    return Skill(
        [np.array([-0.00239468, 0.0078747], dtype=np.float32) for _ in range(length)],
        name="skill_6"
    )