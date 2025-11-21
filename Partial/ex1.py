from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
model = DiscreteBayesianNetwork([
    ('O', 'H'),
    ('O', 'W'),
    ('H', 'R'),
    ('W', 'R'),
    ('H', 'E'),
    ('R', 'C')
])

cpd_O = TabularCPD(variable='O', variable_card=2,
                   values=[[0.3], [0.7]])

cpd_H = TabularCPD(variable='H', variable_card=2,
                   values=[[0.9, 0.2],
                           [0.1, 0.8]],
                   evidence=['O'], evidence_card=[2])

cpd_W = TabularCPD(variable='W', variable_card=2,
                   values=[[0.1, 0.6],
                           [0.9, 0.4]],
                   evidence=['O'], evidence_card=[2])

cpd_R = TabularCPD(variable='R', variable_card=2,
                   values=[
                       [0.6, 0.3, 0.8, 0.2],
                       [0.4, 0.7, 0.2, 0.8]
                   ],
                   evidence=['H', 'W'], evidence_card=[2, 2])

cpd_E = TabularCPD(variable='E', variable_card=2,
                   values=[[0.8, 0.2],
                           [0.2, 0.8]],
                   evidence=['H'], evidence_card=[2])

cpd_C = TabularCPD(variable='C', variable_card=2,
                   values=[[0.85, 0.40],
                           [0.15, 0.60]],
                   evidence=['R'], evidence_card=[2])

model.add_cpds(cpd_O, cpd_H, cpd_W, cpd_R, cpd_E, cpd_C)

assert model.check_model()

plt.figure(figsize=(8,6))
# Construiesc explicit un obiect NetworkX din model pentru a controla desenul
G = nx.DiGraph()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())

pos = nx.spring_layout(G, seed=42)


nx.draw_networkx(G, pos=pos, with_labels=True, node_size=2500, node_color='lightblue', font_size=10, arrows=True)
plt.title("Graf bayes", fontsize=14)
plt.axis('off')
plt.show()

# Inference
inference = VariableElimination(model)

p_H_conditionat_C = inference.query(['H'], evidence={'C': 0})  # C = comfortable
p_E_conditionat_C = inference.query(['E'], evidence={'C': 0})  # C = comfortable
map_query = inference.map_query(['H', 'W'], evidence={'C': 0})  # MAP pentru H, W

print(p_H_conditionat_C)
print(p_E_conditionat_C)
print(f"Map estimate: {map_query}")


# la ce am rezolvat pe foaie de examen




