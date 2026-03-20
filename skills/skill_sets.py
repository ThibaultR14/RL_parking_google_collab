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