import numpy as np
from .skill import Skill

# Avancer
def forward(length=3):
    return Skill([np.array([0.0, 1.0], dtype=np.float32) for _ in range(length)], name="forward")

def forward_left(length=3):
    return Skill([np.array([-0.3, 1.0], dtype=np.float32) for _ in range(length)], name="forward_left")

def forward_right(length=3):
    return Skill([np.array([0.3, 1.0], dtype=np.float32) for _ in range(length)], name="forward_right")

# Reculer
def backward(length=3):
    return Skill([np.array([0.0, -1.0], dtype=np.float32) for _ in range(length)], name="backward")

def backward_left(length=3):
    return Skill([np.array([-0.3, -1.0], dtype=np.float32) for _ in range(length)], name="backward_left")

def backward_right(length=3):
    return Skill([np.array([0.3, -1.0], dtype=np.float32) for _ in range(length)], name="backward_right")
