import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])
sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0, 15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

df = pd.DataFrame({"publicity": publicity, "sales": sales})
print(df)

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
    beta = pm.Normal("beta", mu=0.0, sigma=10.0)
    sigma = pm.HalfNormal("sigma", sigma=10.0)
    
    mu = alpha + beta * publicity
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=sales)
    
    idata = pm.sample(draws=2000, tune=1000, chains=2, target_accept=0.9, return_inferencedata=True, random_seed=42)

print(az.summary(idata, var_names=["alpha", "beta", "sigma"], round_to=4))

hdi_94 = az.hdi(idata, hdi_prob=0.94)[["alpha", "beta", "sigma"]]
hdi_95 = az.hdi(idata, hdi_prob=0.95)[["alpha", "beta", "sigma"]]
print("HDI 94%:\n", hdi_94)
print("HDI 95%:\n", hdi_95)

new_publicity = np.array([2.0, 5.0, 8.0, 11.0])

posterior = idata.posterior.stack(sample=("chain", "draw"))
intercept_samps = posterior["alpha"].values
slope_samps = posterior["beta"].values
sigma_samps = posterior["sigma"].values

n_samples = intercept_samps.shape[0]
preds = np.empty((n_samples, new_publicity.size))
for i in range(n_samples):
    mu_i = intercept_samps[i] + slope_samps[i] * new_publicity
    preds[i, :] = np.random.normal(mu_i, sigma_samps[i])

pred_mean = preds.mean(axis=0)
pred_hdi_low = np.percentile(preds, 3, axis=0)
pred_hdi_high = np.percentile(preds, 97, axis=0)
for x, m, lo, hi in zip(new_publicity, pred_mean, pred_hdi_low, pred_hdi_high):
    print(f"publicity={x}: mean_pred={m:.3f}, HDI94=[{lo:.3f}, {hi:.3f}]")

x_line = np.linspace(publicity.min()-0.5, publicity.max()+0.5, 200)
mean_intercept = intercept_samps.mean()
mean_slope = slope_samps.mean()
y_line = mean_intercept + mean_slope * x_line

plt.figure(figsize=(8, 6))
plt.scatter(publicity, sales)
plt.plot(x_line, y_line)

mu_preds = np.outer(intercept_samps, np.ones_like(x_line)) + np.outer(slope_samps, x_line)
low_mu = np.percentile(mu_preds, 3, axis=0)
high_mu = np.percentile(mu_preds, 97, axis=0)
plt.fill_between(x_line, low_mu, high_mu, alpha=0.25)
plt.xlabel("Publicity")
plt.ylabel("Sales")
plt.title("Bayesian linear regression: mean line and HDI(mu)")
plt.show()
plt.close()
