from .basic_skills import forward, forward_left, forward_right, backward, backward_left, backward_right

def basic_without_backward():
    # Les 3 skills pour avancer
    return [
        forward(3),
        forward_left(3),
        forward_right(3),
    ]

def basic():
    # Les 3 skills pour avancer + 3 skills pour reculer
    return [
        forward(3),
        forward_left(3),
        forward_right(3),
        backward(3),
        backward_left(3),
        backward_right(3),
    ]


from .diayn_4_skills import skill_0, skill_1, skill_2, skill_3

def diayn_parking_4_skills():
    """
    Renvoie la liste des skills découverts par DIAYN pour l'environnement parking-v0.
    Chaque skill correspond à un skill_id du CSV évalué.
    """
    return [
        skill_0(10),  # skill_id 0, length = mean_episode_steps ~20
        skill_1(10),  # skill_id 1
        skill_2(10),  # skill_id 2
        skill_3(10),  # skill_id 3
    ]

from .diayn_5_skills import skill_0, skill_1, skill_2, skill_3, skill_4

def diayn_parking_5_skills():
    """
    Renvoie la liste des 5 skills pour parking-v0 basés sur le CSV.
    Chaque skill a une longueur = mean_episode_steps (200).
    """
    return [
        skill_0(20),  # skill_id 0
        skill_1(20),  # skill_id 1
        skill_2(20),  # skill_id 2
        skill_3(20),  # skill_id 3
        skill_4(20),  # skill_id 4
    ]

from .diayn_6_skills import skill_0, skill_1, skill_2, skill_3, skill_4

def diayn_parking_6_skills():
    """
    Renvoie la liste des 6 skills pour parking-v0 basés sur le CSV.
    Chaque skill a length = 20 steps.
    """
    return [
        skill_0(20),
        skill_1(20),
        skill_2(20),
        skill_3(20),
        skill_4(20),
        skill_5(20),
    ]

from .diayn_7_skills import skill_0, skill_1, skill_2, skill_3, skill_4, skill_5, skill_6

def diayn_parking_7_skills():
    """
    Renvoie la liste des 7 skills pour parking-v0 basés sur le CSV.
    Chaque skill a une longueur = mean_episode_steps.
    """
    return [
        skill_0(20),
        skill_1(20),
        skill_2(20),
        skill_3(20),
        skill_4(20),
        skill_5(20),
        skill_6(20),
    ]

from .diayn_8_skills import skill_0, skill_1, skill_2, skill_3, skill_4, skill_5, skill_6, skill_7

def diayn_parking_8_skills():
    """
    Renvoie la liste des 8 skills pour parking-v0 basés sur le CSV.
    Chaque skill a une longueur = mean_episode_steps.
    """
    return [
        skill_0(20),
        skill_1(20),
        skill_2(20),
        skill_3(20),
        skill_4(20),
        skill_5(20),
        skill_6(20),
        skill_7(20),
    ]