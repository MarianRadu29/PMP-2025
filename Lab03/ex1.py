from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
from itertools import product


def ex():
    # structura grafului Bayesian
    model = DiscreteBayesianNetwork([('S','O'), ('S','L'), ('S','M'), ('L','M')])

    cpd_S = TabularCPD('S', 2, [[0.6], [0.4]])  # randul 0: P(S=0)=0.6, randul 1: P(S=1)=0.4

    # O|S
    cpd_O = TabularCPD('O', 2,
        [[0.9, 0.3],   # P(O=0|S=0)=0.9, P(O=0|S=1)=0.3
        [0.1, 0.7]],  # P(O=1|S=0)=0.1, P(O=1|S=1)=0.7
        evidence=['S'], evidence_card=[2])

    # L|S
    cpd_L = TabularCPD('L', 2,
        [[0.7, 0.2],   # P(L=0|S=0)=0.7, P(L=0|S=1)=0.2
        [0.3, 0.8]],  # P(L=1|S=0)=0.3, P(L=1|S=1)=0.8
        evidence=['S'], evidence_card=[2])

    # M|S,L  (coloanele in ordinea produsului cartezian al evidentelor: (S=0,L=0), (S=0,L=1), (S=1,L=0), (S=1,L=1))
    cpd_M = TabularCPD('M', 2,
        [[0.8, 0.4, 0.5, 0.1],   # P(M=0|S,L)
        [0.2, 0.6, 0.5, 0.9]],  # P(M=1|S,L)  = 0.2,0.6,0.5,0.9 conform enuntului
        evidence=['S','L'], evidence_card=[2,2])

    model.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)
    model.check_model() 


    print("\na) The all independencies:")
    independencies = model.get_independencies()
    for indep in independencies.get_assertions():
        indep_str = str(indep).replace('\u27c2', '_|_')
        print(f"   {indep_str}")

    print("\n\n\nb) Email classification:")

    infer = VariableElimination(model)

    rows = []
    for O, L, M in product([0,1],[0,1],[0,1]):
        q = infer.query(variables=['S'], evidence={'O':O,'L':L,'M':M})
        p_spam = float(q.values[1])        # index 1 corespunde lui S=1
        rows.append({"O":O,"L":L,"M":M,"P(S=1|O,L,M)":round(p_spam,3),
                    "Class": "spam" if p_spam >= 0.5 else "non-spam"})

    df = pd.DataFrame(rows).sort_values(["O","L","M"]).reset_index(drop=True)
    print(df.to_string(index=False))
