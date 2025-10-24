# pip install pgmpy networkx matplotlib numpy
import numpy as np, itertools, matplotlib.pyplot as plt
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import MarkovNetwork
# --- setari & date (5x5, 4 nivele de gri) ---
np.random.seed(0)
H, W, L = 5, 5, 4
lam, beta = 2.0, 1.0
noise_rate = 0.10

img_true  = np.random.randint(0, L, (H, W))
img_noisy = img_true.copy()
k = max(1, int(noise_rate * H * W))
for u in np.random.choice(H*W, k, replace=False):
    r, c = divmod(u, W)
    img_noisy[r, c] = (img_noisy[r, c] + 1 + np.random.randint(0, L-1)) % L

# --- (a) MRF pe grila 4-neighbors ---
vid = lambda r,c: f"X_{r}_{c}"
vars_all = [vid(r,c) for r in range(H) for c in range(W)]
G = MarkovNetwork()
G.add_nodes_from(vars_all)
edges = [(vid(r,c), vid(r, c+1)) for r in range(H) for c in range(W-1)] + \
        [(vid(r,c), vid(r+1, c)) for r in range(H-1) for c in range(W)]
G.add_edges_from(edges)

# factori: Phi_i = exp(-λ(x_i - y_i)^2), Phi_ij = exp(-β(x_i - x_j)^2)
levels = np.arange(L)
pair_vals = np.exp(-beta * (levels[:,None] - levels[None,:])**2)
factors = []
for r in range(H):
    for c in range(W):
        y = img_noisy[r,c]
        uni = np.exp(-lam * (levels - y)**2)
        factors.append(DiscreteFactor([vid(r,c)], [L], uni))
for u,v in edges:
    factors.append(DiscreteFactor([u,v], [L,L], pair_vals))

G.add_factors(*factors)
G.check_model()          

# --- (b) MAP cu BeliefPropagation ---
bp = BeliefPropagation(G)
assign = bp.map_query(variables=vars_all)            # dict: var -> {0..L-1}
img_map = np.array([[assign[vid(r,c)] for c in range(W)] for r in range(H)])

# --- metrica + vizualizare compacta ---
acc_noisy = (img_noisy == img_true).mean()
acc_map   = (img_map   == img_true).mean()
print(f"Pixel accuracy: noisy={acc_noisy:.2%} | MAP={acc_map:.2%}")

for t, im in [("Original", img_true), ("Noisy", img_noisy), ("MAP", img_map)]:
    plt.figure(figsize=(2.2,2.2)); plt.imshow(im, vmin=0, vmax=L-1); plt.title(t); plt.axis('off')
plt.show()