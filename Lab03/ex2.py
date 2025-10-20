from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Variabile:
# D (Die/Zar): 0=prim(2,3,5), 1=6, 2=altfel(1,4)
# B (Ball/Bila): 0=rosie, 1=albastra, 2=neagra

# Structura: D -> B (rezultatul zarului influenteaza bila extrasa)
model = DiscreteBayesianNetwork([('D', 'B')])

# P(D): probabilitati pentru zar
# D=0 (prim: 2,3,5): 3/6 = 0.5
# D=1 (6): 1/6 = 0.167
# D=2 (altfel: 1,4): 2/6 = 0.333
cpd_D = TabularCPD('D', 3, [[0.5], [0.167], [0.333]])

# P(B|D): probabilitatea bilei in functie de zar
# Initial: 3 rosii, 4 albastre, 2 negre (total 9 bile)

# Daca D=0 (prim -> adaug neagra): 3R, 4A, 3N (total 10)
#   P(B=0|D=0) = 3/10 = 0.3 (rosie)
#   P(B=1|D=0) = 4/10 = 0.4 (albastra)
#   P(B=2|D=0) = 3/10 = 0.3 (neagra)

# Daca D=1 (6 -> adaug rosie): 4R, 4A, 2N (total 10)
#   P(B=0|D=1) = 4/10 = 0.4 (rosie)
#   P(B=1|D=1) = 4/10 = 0.4 (albastra)
#   P(B=2|D=1) = 2/10 = 0.2 (neagra)

# Daca D=2 (altfel -> adaug albastra): 3R, 5A, 2N (total 10)
#   P(B=0|D=2) = 3/10 = 0.3 (rosie)
#   P(B=1|D=2) = 5/10 = 0.5 (albastra)
#   P(B=2|D=2) = 2/10 = 0.2 (neagra)

cpd_B = TabularCPD('B', 3,
    [[0.3, 0.4, 0.3],   # P(B=0|D) - rosie
     [0.4, 0.4, 0.5],   # P(B=1|D) - albastra
     [0.3, 0.2, 0.2]],  # P(B=2|D) - neagra
    evidence=['D'], evidence_card=[3])

model.add_cpds(cpd_D, cpd_B)
model.check_model()


infer = VariableElimination(model)
result = infer.query(variables=['B'])

print(f"b) P(bila rosie) = {result.values[0]:.4f}")

# acelasi rezultat ca in laboratorul anterior (0.3167)
