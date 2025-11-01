from dataclasses import dataclass
from typing import List, Tuple, Sequence, Union
import numpy as np

@dataclass
class HMM:
    states: List[str]             # ex: ["D","M","E"]
    observations: List[str]       # ex: ["FB","B","S","NS"]
    startprob: np.ndarray         # pi  (shape: N,)
    transmat: np.ndarray          # A  (shape: N,N)
    emissionprob: np.ndarray      # B  (shape: N,M)

def viterbi(hmm: HMM, obs_seq: Sequence[Union[str, int]]) -> Tuple[float, List[str]]:
    """
    Pot accepta ca parametru `obs_seq` fie ca o secventa de etichete (ex. "FB"), fie ca indici (int).
    """
    # map observatii -> indici
    obs_to_idx = {o: i for i, o in enumerate(hmm.observations)}
    O = np.array([obs_to_idx[o] if isinstance(o, str) else int(o) for o in obs_seq], dtype=int)

    N = len(hmm.states)
    T = len(O)

    # log sigur (log(0) = -infinit)
    def safe_log(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.full_like(x, -np.inf)
        mask = x > 0
        out[mask] = np.log(x[mask])
        return out

    log_pi = safe_log(hmm.startprob)       # (N,)
    log_A  = safe_log(hmm.transmat)        # (N,N)
    log_B  = safe_log(hmm.emissionprob)    # (N,M)

    # DP
    delta = np.full((T, N), -np.inf)
    psi   = np.full((T, N), -1, dtype=int)

    # initializare
    delta[0, :] = log_pi + log_B[:, O[0]]

    # recurenta
    for t in range(1, T):
        for j in range(N):
            scores = delta[t-1, :] + log_A[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + log_B[j, O[t]]

    # terminare
    last_state = int(np.argmax(delta[T-1, :]))
    best_logprob = float(delta[T-1, last_state])

    # backtrack
    path_idx = np.empty(T, dtype=int)
    path_idx[-1] = last_state
    for t in range(T-1, 0, -1):
        path_idx[t-1] = psi[t, path_idx[t]]

    best_path = [hmm.states[i] for i in path_idx.tolist()]
    return float(np.exp(best_logprob)), best_path
