import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('date_promovare_examen.csv')

    counts = df['Promovare'].value_counts()
    print(counts)
    if counts[0] == counts[1]:
        print("=> Datele sunt perfect balansate\n")
    else:
        print("=> Datele NU sunt balansate\n")

    # load date
    X = df[['Ore_Studiu', 'Ore_Somn']].values
    y = df['Promovare'].values

    # standardizare date
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    print("Construim modelul PyMC...")
    with pm.Model():
        # Priori
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=2)

        # regresia logistica
        mu = alpha + pm.math.dot(X_scaled, beta)
        theta = pm.math.sigmoid(mu)

        # granita de decizie
        bd = pm.Deterministic('bd', -alpha/beta[1] - (beta[0]/beta[1] * X_scaled[:, 0]))

        # Likelihood
        y_obs = pm.Bernoulli('y_obs', p=theta, observed=y)

        idata = pm.sample(2000, return_inferencedata=True, progressbar=True)

    # granita de decizie si separarea datelor
    summary = az.summary(idata, var_names=['alpha', 'beta'])
    print("\n\n\nRezumatul parametrilor (pe date scalate):")
    print(summary)

    # Extragem mediile parametrilor
    alpha_mean = summary.loc['alpha', 'mean']
    beta_studiu_mean = summary.loc['beta[0]', 'mean']
    beta_somn_mean = summary.loc['beta[1]', 'mean']

    # Putem calcula acuratetea folosind mediile parametrilor
    mu_pred = alpha_mean + np.dot(X_scaled, [beta_studiu_mean, beta_somn_mean])
    y_pred = (1 / (1 + np.exp(-mu_pred)) > 0.5).astype(int)
    acc = np.mean(y_pred == y)
    print(f"\n\n\nb) Acuratetea modelului pe datele de antrenare: {acc:.2f} ({(acc*100):.0f}%)")
    if acc == 1.0:
        print("=> Datele sunt BINE SEPARATE (chiar perfect separate liniar).\n")
    else:
        print("=> Datele au o zona de suprapunere(au zgomot).\n")

    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Picat (0)')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Promovat (1)')

    # revenim la scara originala pentru plot
    x_vals = np.array([X[:, 0].min(), X[:, 0].max()])

    x_vals_scaled = (x_vals - X_mean[0]) / X_std[0]
    y_vals_scaled = -(alpha_mean + beta_studiu_mean * x_vals_scaled) / beta_somn_mean
    y_vals = y_vals_scaled * X_std[1] + X_mean[1]

    plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Granita de Decizie')
    plt.xlabel('Ore Studiu')
    plt.ylabel('Ore Somn')
    plt.legend()
    plt.title(f'regresie logistica (acc: {acc:.2f})')
    plt.show()

    print("c)")
    print(f"Coeficient 'Ore Studiu' (mediu, scalat): {beta_studiu_mean:.4f}")
    print(f"Coeficient 'Ore Somn'   (mediu, scalat): {beta_somn_mean:.4f}")

    if abs(beta_somn_mean) > abs(beta_studiu_mean):
        print("CONCLUZIE: 'Orele de SOMN' influenteaza mai mult promovabilitatea.")
    else:
        print("CONCLUZIE: 'Orele de STUDIU' influenteaza mai mult promovabilitatea.")


if __name__ == "__main__":
    main()