import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

df = pd.read_csv('Prices.csv')
y_val = df['Price'].values
x1_val = df['Speed'].values
x2_val = np.log(df['HardDrive'].values)  # Logaritm natural conform cerinței

# a)
with pm.Model() as model:
    # MutableData pentru predictii ulterioare
    x1_shared = pm.Data("x1_shared", x1_val)
    x2_shared = pm.Data("x2_shared", x2_val)
    
    # Priori slab informative
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=100)
    
    mu = alpha + beta1 * x1_shared + beta2 * x2_shared
    y_pred = pm.Normal('y_pred', mu=mu, sigma=sigma, observed=y_val)
    
    idata = pm.sample(2000, chains=2, return_inferencedata=True, progressbar=False)

# Rezultate punctele b) si c)
print("Rezultate b) HDI 95%")
summary = az.summary(idata, var_names=['beta1', 'beta2'], hdi_prob=0.95, kind='stats')
print(summary[['mean', 'hdi_2.5%', 'hdi_97.5%']])

# Verificare utilitate predictori
b1_hdi = az.hdi(idata.posterior['beta1'], hdi_prob=0.95).to_numpy().flatten()
b2_hdi = az.hdi(idata.posterior['beta2'], hdi_prob=0.95).to_numpy().flatten()
print("\nRezultate c) Interpretare")
print(f"Beta1 (Speed) util? {'DA' if (b1_hdi[0] > 0 or b1_hdi[1] < 0) else 'NU'}")
print(f"Beta2 (HD) util?    {'DA' if (b2_hdi[0] > 0 or b2_hdi[1] < 0) else 'NU'}")

# Predictii pentru Speed=33, HardDrive=540 (Punctele d si e)
new_x1 = [33]
new_x2 = [np.log(540)]

with model:
    pm.set_data({"x1_shared": new_x1, "x2_shared": new_x2})
    ppc = pm.sample_posterior_predictive(idata, predictions=True, progressbar=False)

# d) Pretul asteptat (mu) - calcul manual din trace
post = idata.posterior
mu_dist = post['alpha'] + post['beta1'] * 33 + post['beta2'] * np.log(540)
mu_hdi = az.hdi(mu_dist, hdi_prob=0.90).to_numpy().flatten()

# e) Pretul predictiv (y) - din posterior predictive
y_pred_dist = ppc.predictions['y_pred']
y_hdi = az.hdi(y_pred_dist, hdi_prob=0.90).to_numpy().flatten()

print("\n Rezultate d) & e) HDI 90% pentru Speed=33, HardDrive=540")
print(f"Pret asteptat (mu): [{mu_hdi[0]:.1f}, {mu_hdi[1]:.1f}]")
print(f"Pret predictie (y): [{y_hdi[0]:.1f}, {y_hdi[1]:.1f}]")

# bonus: variabila Premium
print("\n Bonus: Influenta Premium ")
is_premium = (df['premium'] == 'yes').astype(int).values

with pm.Model() as model_bonus:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    b1 = pm.Normal('beta1', mu=0, sigma=10)
    b2 = pm.Normal('beta2', mu=0, sigma=10)
    b_prem = pm.Normal('beta_premium', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=100)
    
    mu = alpha + b1*x1_val + b2*x2_val + b_prem*is_premium
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=y_val)
    
    idata_bonus = pm.sample(1000, chains=2, progressbar=False)

bp_hdi = az.hdi(idata_bonus.posterior['beta_premium'], hdi_prob=0.95).to_numpy().flatten()
print(f"Beta Premium HDI 95%: [{bp_hdi[0]:.2f}, {bp_hdi[1]:.2f}]")
print(f"Concluzie: {'Premium influenteaza pretul' if (bp_hdi[0] > 0 or bp_hdi[1] < 0) else 'Nu influenteaza clar'}")
