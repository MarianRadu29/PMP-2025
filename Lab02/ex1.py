import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def simulate_experiment():
    urn = np.array(['R'] * 3 + ['A'] * 4 + ['N'] * 2)
    
    die = np.random.randint(1, 7)
    
    if die in [2, 3, 5]:        
        urn.append('N')
    elif die == 6:
        urn.append('R')
    else:                     
        urn.append('A')
    
    extracted_ball = np.random.choice(np.random.shuffle(urn))
    return extracted_ball


def estimate_red_probability(n=100000):
    count_red = 0
    for _ in range(n):
        if simulate_experiment() == 'R':
            count_red += 1
    return count_red / n


def ex1():
    p_empiric = estimate_red_probability()
    print(f"The probability for extraction the red ball: {p_empiric:.4f}")
    #  - Prime: 2, 3, 5 → adaug negru (3 cazuri)
    #  - 6 → adaug rosu (1 caz)
    #  - 1, 4 → adaug albastru (2 cazuri)


    # numarator -> nr bile rosii
    # suma numitor -> count (black balls + red balls + blue balls)

    p_red_black = 3 / (3 + 4 + 3)

    p_red_red = 4 / (4 + 4 + 2)

    p_red_blue = 3 / (3 + 5 + 2)

    p_teoretic = (3/6) * p_red_black + (1/6) * p_red_red + (2/6) * p_red_blue

    print(f"Theoretical probability for extract a red ball is: {p_teoretic:.4f}")
    print(f"The difference between estimation and theoretical is: {abs(p_teoretic - p_empiric):.4f}")