import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from matplotlib import patheffects
import networkx as nx

states = ["D", "M", "E"]  # Difficult, Medium, Easy
state_full = {"D": "Difficult", "M": "Medium", "E": "Easy"}

grades = ["FB", "B", "S", "NS"]
grade_to_idx = {g:i for i,g in enumerate(grades)}

# matricea de tranzitie( liniile: starea curenta, coloanele: starea urmatoare)
transition_matrix = np.array([
    [0.0, 0.5, 0.5],   # D -> (M,E)
    [0.5, 0.25, 0.25], # M -> (D,M,E)
    [0.5, 0.25, 0.25]  # E -> (D,M,E)
], dtype=float)

# distributia initiala pi
pi = np.array([1/3, 1/3, 1/3], dtype=float)

# emisia probabilitatilor B ( randuri: stari D,M,E; coloane: observatii FB,B,S,NS)
emission_prob = np.array([
    [0.10, 0.20, 0.40, 0.30],  # D
    [0.15, 0.25, 0.50, 0.10],  # M
    [0.20, 0.30, 0.40, 0.10],  # E
], dtype=float)

# secventa observata (notele primite)
obs_seq = ["FB","FB","S","B","B","S","B","B","NS","B","B"]
O = np.array([grade_to_idx[g] for g in obs_seq], dtype=int)
len_obs_seq = len(O)

model = hmm.CategoricalHMM(n_components=3, init_params="")
model.startprob_ = pi.copy()
model.transmat_ = transition_matrix.copy()
model.emissionprob_ = emission_prob.copy()

# (shape: T x 1)
# pentru fiecare element din O, creeaza o linie in X
observations_sequence = O.reshape(-1, 1)


# b) Probabilitatea secventei de observatii
logP = model.score(observations_sequence) # returneaza logaritmul probabilitatii
P = np.exp(logP)
print(f"   Secventa observata: {obs_seq}")
print(f"   P(O) = {P:.10f}")
print("\n\n\n")

# c) Secventa cea mai probabila de dificultati (algoritmul Viterbi)
logprob_viterbi, best_path = model.decode(observations_sequence, algorithm="viterbi")
prob_viterbi = np.exp(logprob_viterbi)

# Convertim indicii in etichete de stari
best_path_labels = [states[i] for i in best_path]

print(f"   Observatii:    {obs_seq}")
print(f"   Stari:         {best_path_labels}")
print(f"   P(secventa optima) = {prob_viterbi:.10f}")
print("\n\n\n")






# a) Diagrama de stari 
G = nx.DiGraph()
G.add_nodes_from(states)

# Adaugare muchii cu probabilitati
for i, si in enumerate(states):
    for j, sj in enumerate(states):
        if transition_matrix[i, j] > 0:
            G.add_edge(si, sj, weight=transition_matrix[i, j])

# Vizualizare
plt.figure(figsize=(10, 8))

# Pozitii custom pentru noduri (triunghi)
pos = {"D": (0.5, 1), "M": (0, 0), "E": (1, 0)}

# Desenare noduri
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='lightblue', 
                       node_shape='o', edgecolors='black', linewidths=2)

# Desenare etichete noduri
nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold')

# Desenare muchii - fiecare muchie separat cu arc pentru a vedea ambele directii
# muchiile cu starile neobservabile ar trebui desenate si ele
import matplotlib.patches as mpatches
ax = plt.gca()

for i, si in enumerate(states):
    for j, sj in enumerate(states):
        if transition_matrix[i, j] > 0:
            start_pos = np.array(pos[si])
            end_pos = np.array(pos[sj])
            
            # Verificam daca exista si muchia inversa pentru a folosi arce diferite
            if transition_matrix[j, i] > 0 and i < j:  # Desenam doar o data perechea
                # Muchii bidirectionale - doua arce cu raze opuse
                # Arc de la si la sj
                nx.draw_networkx_edges(G, pos, [(si, sj)], width=2.5, arrowsize=20, 
                                      arrows=True, edge_color='gray', 
                                      connectionstyle='arc3,rad=0.3')
                # Arc de la sj la si (cu raza negativa pentru a curba in partea opusa)
                nx.draw_networkx_edges(G, pos, [(sj, si)], width=2.5, arrowsize=20, 
                                      arrows=True, edge_color='gray', 
                                      connectionstyle='arc3,rad=0.3')
            elif transition_matrix[j, i] == 0:
                # Muchie unidirectionala
                nx.draw_networkx_edges(G, pos, [(si, sj)], width=2.5, arrowsize=20, 
                                      arrows=True, edge_color='gray', 
                                      connectionstyle='arc3,rad=0.2')


for i, si in enumerate(states):
    for j, sj in enumerate(states):
        if transition_matrix[i, j] > 0:
            start_pos = np.array(pos[si])
            end_pos = np.array(pos[sj])
            
            # Calculam pozitia etichetei pe arc
            mid_pos = (start_pos + end_pos) / 2
            
            # Offset perpendicular pentru a pozitiona eticheta pe arc
            vec = end_pos - start_pos
            perp = np.array([-vec[1], vec[0]])
            perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else perp
            
            # Daca este bidirectionala, offset in directii opuse
            if transition_matrix[j, i] > 0:
                # Offset pozitiv pentru muchia de la i la j
                label_pos = mid_pos + perp * 0.15
            else:
                # Centrat pentru muchie unidirectionala
                label_pos = mid_pos + perp * 0.1
            
            # Desenare eticheta
            txt = plt.text(label_pos[0], label_pos[1], f"{transition_matrix[i, j]:.2f}",
                          fontsize=11, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                   edgecolor='gray', alpha=0.9))
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])

plt.title("a) HMM State Diagram\n(D=Difficult, M=Medium, E=Easy)", 
          fontsize=14, fontweight='bold', pad=20)
plt.axis('off')
plt.tight_layout()
plt.savefig("Lab05/state_diagram.png", dpi=150, bbox_inches='tight')
# plt.show()
