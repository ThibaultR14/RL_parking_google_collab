import numpy as np
from .skill import Skill

# Skill 0
def skill_0(length=20):
    return Skill(
        [np.array([-0.01068813, 0.06109793], dtype=np.float32) for _ in range(length)],
        name="skill_0"
    )

# Skill 1
def skill_1(length=20):
    return Skill(
        [np.array([0.00589651, 0.01104033], dtype=np.float32) for _ in range(length)],
        name="skill_1"
    )

# Skill 2
def skill_2(length=20):
    return Skill(
        [np.array([-0.00079506, -0.02607705], dtype=np.float32) for _ in range(length)],
        name="skill_2"
    )

# Skill 3
def skill_3(length=20):
    return Skill(
        [np.array([0.01870886, -0.00950817], dtype=np.float32) for _ in range(length)],
        name="skill_3"
    )

# Skill 4
def skill_4(length=20):
    return Skill(
        [np.array([0.05757369, -0.04174677], dtype=np.float32) for _ in range(length)],
        name="skill_4"
    )

# Skill 5
def skill_5(length=20):
    return Skill(
        [np.array([-0.00066416, -0.02294944], dtype=np.float32) for _ in range(length)],
        name="skill_5"
    )

# Skill 6
def skill_6(length=20):
    return Skill(
        [np.array([-0.0129176, -0.06412915], dtype=np.float32) for _ in range(length)],
        name="skill_6"
    )

# Skill 7
def skill_7(length=20):
    return Skill(
        [np.array([-0.00446738, 0.03770607], dtype=np.float32) for _ in range(length)],
        name="skill_7"
    )