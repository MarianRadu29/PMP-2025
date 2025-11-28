# exercise1_pymc_ro_fixed.py
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path

np.random.seed(42)

#  Configurare
prior_mu = 10.0
Ys = [0, 5, 10]
thetas = [0.2, 0.5]
scenarii = [(y, th) for y, th in product(Ys, thetas)]
n_draws = 4000
n_tune = 2000
chains = 4
target_accept = 0.9  # folosit de NUTS daca apar variabile continue, altfel inofensiv

# Folder pentru grafice
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

# Stocare rezultate
posterior_ns = {}
ppc_Ystars = {}

# Model si esantionare
for (y_obs, theta) in scenarii:
    with pm.Model() as model:
        # Prior: n aprox Poisson
        n = pm.Poisson("n", mu=prior_mu)
        # Verosimilitate: Y | n, theta aprox Binomial(n, theta)
        y = pm.Binomial("Y", n=n, p=theta, observed=y_obs)

        # Sampler pentru variabile discrete
        step = pm.Metropolis()
        idata = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=chains,
            step=step,
            random_seed=42,
            progressbar=True,
            cores=1,
            return_inferencedata=True,
        )

        # Predictiv posterior pentru un Y* viitor
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["Y"],
            random_seed=123,
            extend_inferencedata=True,
        )

    posterior_ns[(y_obs, theta)] = idata
    ppc_Ystars[(y_obs, theta)] = ppc

# (a) Posterior n: 3x2 (6 scenarii)
rows, cols = 3, 2
fig, axes = plt.subplots(rows, cols, figsize=(12, 14))
axes = axes.ravel()

for ax, (y_obs, theta) in zip(axes, scenarii):
    idata = posterior_ns[(y_obs, theta)]
    az.plot_posterior(
        idata,
        var_names=["n"],
        point_estimate="mean",
        hdi_prob=0.94,
        ax=ax,
    )
    ax.set_title(f"Posterior pentru n | Y={y_obs}, θ={theta}")

# Daca raman axe nefolosite (in caz ca scenariile nu umplu grila fixata), le ascundem
for k in range(len(scenarii), len(axes)):
    axes[k].axis("off")

plt.tight_layout()
post_path = fig_dir / "posterior_n.png"
fig.savefig(post_path, dpi=150)
plt.close(fig)

# ---- (c) Posterior predictiv Y*: 3x2 (6 scenarii) ----
fig2, axes2 = plt.subplots(rows, cols, figsize=(12, 14))
axes2 = axes2.ravel()

for ax, (y_obs, theta) in zip(axes2, scenarii):
    ppc = ppc_Ystars[(y_obs, theta)]
    ystar = ppc.posterior_predictive["Y"].values.flatten()
    az.plot_dist(ystar, ax=ax)
    ax.set_xlabel("Y*")
    ax.set_title(f"Predictiv posterior Y* | Y={y_obs}, θ={theta}")

for k in range(len(scenarii), len(axes2)):
    axes2[k].axis("off")

plt.tight_layout()
ppc_path = fig_dir / "predictive_y_star.png"
fig2.savefig(ppc_path, dpi=150)
plt.close(fig2)

print(f"Salvate: {post_path} si {ppc_path}")
