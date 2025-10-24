 
import numpy as np
from math import comb
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# 1)
rng = np.random.default_rng(0)
p_rigged, p_fair = 4/7, 1/2

def game():
    starter = rng.integers(0, 2)   # 0=P0, 1=P1
    n = rng.integers(1, 7)
    if starter == 0:
        m = rng.binomial(2*n, p_rigged)
    else:
        m = rng.binomial(2*n, p_fair)
    return (starter == 0 and n >= m) or (starter == 1 and n < m)

p0 = sum(game() for _ in range(10_000))/10_000
print(f"P0 win aprox. {p0:.3f}, P1 win aprox. {1-p0:.3f}")


# 2)

model = DiscreteBayesianNetwork([('S','M'), ('N','M')])

cpd_S = TabularCPD('S', 2, [[0.5],[0.5]], state_names={'S':['P0','P1']})
cpd_N = TabularCPD('N', 6, [[1/6]]*6, state_names={'N':[1,2,3,4,5,6]})

m_vals = range(13)
cols = []
for s,p in [('P0',4/7), ('P1',1/2)]:
    for n in range(1,7):
        probs = [comb(2*n,m)*(p**m)*((1-p)**(2*n-m)) if m<=2*n else 0 for m in m_vals]
        cols.append(probs)

cpd_M = TabularCPD('M', 13, list(map(list, zip(*cols))),
                   evidence=['S','N'], evidence_card=[2,6],
                   state_names={'S':['P0','P1'],'N':[1,2,3,4,5,6],'M':list(m_vals)})

model.add_cpds(cpd_S, cpd_N, cpd_M)
model.check_model()


# 3)
def P_m1(p):
    return sum(comb(2*n,1)*p*(1-p)**(2*n-1) for n in range(1,7))/6

L_P0, L_P1 = P_m1(4/7), P_m1(1/2)
post_P0 = 0.5*L_P0/(0.5*L_P0 + 0.5*L_P1)
print(f"P(S=P0|m=1)={post_P0:.3f}, P(S=P1|m=1)={1-post_P0:.3f}")