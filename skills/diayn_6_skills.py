import numpy as np
from .skill import Skill

# Skill 0
def skill_0(length=20):
    return Skill(
        [np.array([-0.03832629, 0.08111883], dtype=np.float32) for _ in range(length)],
        name="skill_0"
    )

# Skill 1
def skill_1(length=20):
    return Skill(
        [np.array([-0.0584769, 0.05069781], dtype=np.float32) for _ in range(length)],
        name="skill_1"
    )

# Skill 2
def skill_2(length=20):
    return Skill(
        [np.array([0.04894267, -0.05108061], dtype=np.float32) for _ in range(length)],
        name="skill_2"
    )

# Skill 3
def skill_3(length=20):
    return Skill(
        [np.array([-0.00546154, 0.01033197], dtype=np.float32) for _ in range(length)],
        name="skill_3"
    )

# Skill 4
def skill_4(length=20):
    return Skill(
        [np.array([0.00389242, 0.00020639], dtype=np.float32) for _ in range(length)],
        name="skill_4"
    )

# Skill 5
def skill_5(length=20):
    return Skill(
        [np.array([-0.10948912, -0.03697768], dtype=np.float32) for _ in range(length)],
        name="skill_5"
    )