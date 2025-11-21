import numpy as np
from hmmlearn import hmm


states = ["W", "R", "S"]          # Walking, Running, Resting
observations = ["L", "M", "H"]    # Low, Medium, High

n_states = len(states)
n_observations = len(observations)

start_prob = np.array([0.4, 0.3, 0.3])

trans_prob = np.array([
    [0.6, 0.3, 0.1],  # W -> W,R,S
    [0.2, 0.7, 0.1],  # R -> W,R,S
    [0.3, 0.2, 0.5]   # S -> W,R,S
])

emit_prob = np.array([
    [0.1, 0.7, 0.2],   # W -> L,M,H
    [0.05, 0.25, 0.7], # R -> L,M,H
    [0.8, 0.15, 0.05]  # S -> L,M,H
])

# convertesc obs in indici
obs_map = {obs: i for i, obs in enumerate(observations)}

# SECVENTA OBSERVATA DE LA SUBPUNCTUL B
obs_sequence = [obs_map[o] for o in ["M", "H", "L"]]

X_counts = np.eye(n_observations)[obs_sequence]  # shape (T, n_observations)

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_prob
model.transmat_ = trans_prob
model.emissionprob_ = emit_prob
# setam n_trials (cate incercari pe pas); aici 1 pentru fiecare pas de timp
model.n_trials = np.ones(X_counts.shape[0], dtype=int)

# Algoritmul Forward: probabilitatea secventei observate (folosind conturi)
logprob = model.score(X_counts)
prob = np.exp(logprob)
print(f"Probabilitatea secventei observate [M,H,L]: {prob:.6f}")

# Algoritmul Viterbi: cea mai probabila secventa de stari ascunse
logprob_vit, state_sequence = model.decode(X_counts, algorithm="viterbi")
state_sequence_names = [states[i] for i in state_sequence]
print(f"Cea mai probabila secventa de stari ascunse: {state_sequence_names}")

# Generarea a 10.000 sequence si estimarea probabilitatii empirice
n_samples = 10000
count_match = 0

for _ in range(n_samples):
    Xs, Z = model.sample(len(obs_sequence))
    sampled_indices = list(Xs.argmax(axis=1))
    if sampled_indices == obs_sequence:
        count_match += 1

empirical_prob = count_match / n_samples
print("\n\n\n")

print(f"Probabilitatea empirica pentru secventa [M,H,L]: {empirical_prob:.6f}")

print(f"Probabilitate Forward algorithm: {prob:.6f}")


# la punctul C am scris pe foaie rezolvarea