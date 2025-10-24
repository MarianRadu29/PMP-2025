# pip install pgmpy networkx matplotlib
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import itertools, numpy as np
import networkx as nx, matplotlib.pyplot as plt

G = MarkovNetwork()
G.add_nodes_from(["A1","A2","A3","A4","A5"])
G.add_edges_from([("A1","A2"),("A1","A3"),("A2","A4"),("A2","A5"),("A3","A4"),("A4","A5")])
print(G.edges())
print("Maximal cliques:", list(nx.find_cliques(G)))


# Phi(Ai1,...,Aik) = exp(i1*Ai1 + ... + ik*Aik), cu Ai ∈ {-1,+1} ---
def make_factor(vars_list, coeffs):
    vals = []
    for state01 in itertools.product([0,1], repeat=len(vars_list)):   # 0→-1, 1→+1
        spins = [(-1 if s==0 else 1) for s in state01]
        vals.append(np.exp(sum(c*a for c,a in zip(coeffs, spins))))
    return DiscreteFactor(vars_list, [2]*len(vars_list), np.array(vals, float))

# Clique-uri maxime & potentiale:
phi_12   = make_factor(["A1","A2"],      [1,2])
phi_13   = make_factor(["A1","A3"],      [1,3])
phi_34   = make_factor(["A3","A4"],      [3,4])
phi_245  = make_factor(["A2","A4","A5"], [2,4,5])
G.add_factors(phi_12, phi_13, phi_34, phi_245)  
G.check_model()

# --- (b) Joint ne-normalizat, Z, probabilitati, MAP ---
joint = phi_12 * phi_13 * phi_34 * phi_245
Z = joint.values.sum()
print(f"Z = {Z:.6f}")

# MAP prin enumerare pe joint
imax = int(np.argmax(joint.values))
assign01 = np.array(np.unravel_index(imax, tuple(joint.cardinality)))
assign_pm1 = {v: (-1 if s==0 else 1) for v,s in zip(joint.variables, assign01)}
p_map = float(joint.values.flat[imax] / Z)
print("MAP (enumerare):", assign_pm1, f"with P={p_map:.6f}")

# MAP si cu BP (confirmare)
bp = BeliefPropagation(G)
map_bp = bp.map_query(variables=list(G.nodes()), show_progress=False)
map_bp_pm1 = {v: (-1 if s==0 else 1) for v,s in map_bp.items()}
print("MAP (BeliefPropagation):", map_bp_pm1)
