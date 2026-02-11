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
