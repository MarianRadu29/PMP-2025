import numpy as np
import matplotlib.pyplot as plt

def ex1():
    n = 1000

    lambdas = np.array([1, 2, 5, 10])

    data = {}
    for lam in lambdas:
        data[lam] = np.random.poisson(lam, n)

    for (key, values) in data.items():
        plt.hist(values, bins=np.arange(0, max(values) + 2) - 0.5, 
                 color='red', edgecolor='black', density=True)
        plt.xlabel("Numar apeluri (k)")
        plt.ylabel("Frecventa relativa / densitate")
        plt.title(f"Distributia Poisson(λ={key}) - {n} simulari")
        plt.xticks(range(0, max(values) + 1))
        plt.tight_layout()
        plt.show()


def ex2():
    np.random.seed(42)
    n = 1000
    lambdas = np.array([1, 2, 5, 10])

    # alegem aleator lambda din {1,2,5,10} cu probabilitate egala
    random_lambdas = np.random.choice(lambdas, size=n, replace=True)

    # generez o valoare Poisson corespunzatoare fiecarui lambda
    random_poisson_values = np.random.poisson(lam=random_lambdas)

    # histograma distributiei randomizate
    plt.hist(random_poisson_values, 
             bins=np.arange(0, max(random_poisson_values) + 2) - 0.5, 
             color='red', edgecolor='black', density=True)
    plt.xlabel("Numar apeluri (k)")
    plt.ylabel("Frecventa relativa / densitate")
    plt.title("Distributia Poisson randomizata (lambda = {1,2,5,10}, uniform) - 1000 simulari")
    plt.xticks(range(0, max(random_poisson_values) + 1))
    plt.tight_layout()
    plt.show()


ex1()
ex2()

# # b)

# How does the shape of the randomized distribution differ from the fixed ones?
# Distributiile Poisson cu lambda fix au o forma clara, cu varianta aproximativ egala cu media.
# Distributia randomizata este o mixtura de mai multe Poisson-uri si are o varianta mai mare decat media (over-dispersion).

# What does this tell you about the effect of parameter uncertainty or variability in modeling real-world processes?
# Aceasta arata ca in procesele reale, unde rata lambda variaza (de exemplu, ore mai aglomerate si ore mai linistite), modelarea cu un lambda fix subestimeaza variabilitatea datelor.
# Prin urmare, luarea in considerare a incertitudinii parametrului lambda face modelul mai realist.
