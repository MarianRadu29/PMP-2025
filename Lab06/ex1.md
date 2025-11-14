# Exercitiul 1

## Datele problemei
- Prevalența bolii: $P(B) = 0.01$
- Sensibilitatea: $P(+|B) = 0.95$
- Specificitatea: $P(-|\neg B) = 0.90$

## a) Probabilitatea ca persoana sa aiba boala daca testul iese pozitiv
Aplic teorema lui Bayes:

$$
P(B|+) = \frac{P(+|B)P(B)}{P(+|B)P(B) + P(+|\neg B)P(\neg B)}
$$

$$
P(+|\neg B) = 1 - \text{specificitate} = 1 - 0.90 = 0.10
$$

$$
P(B|+) = \frac{0.95 \cdot 0.01}{0.95 \cdot 0.01 + 0.10 \cdot 0.99} = 0.08756
$$

$$
\boxed{P(B|+) \approx 8.76\%}
$$

---

## b) Specificitatea minima pentru care $P(B|+) = 0.5$

Din teorema lui Bayes:

$$
0.5 = \frac{P(+|B)P(B)}{P(+|B)P(B) + (1 - \text{spec})(1 - P(B))}
$$

Rezolv dupa specificitate:

$$
\text{spec}_{\min} = 1 - \frac{P(+|B)P(B)}{1 - P(B)} = 1 - \frac{0.95 \cdot 0.01}{0.99} = 0.9904
$$

$$
\boxed{\text{spec}_{\min} \approx 99.04\%}
$$

## Rezumat

| Parametru                                    | Valoare |
|----------------------------------------------|---------|
| P(B\|+) (la specificitate = 0.90)           | 8.76%   |
| Specificitate minimă pentru P(B\|+) = 0.5   | 99.04%  |

---

## Cod Python
```python
prev = 0.01      # P(B)
sens = 0.95      # P(+|B) 
spec = 0.90      # P(-| not B) 

# a) 
post = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
print("a) P(B | +) =", round(post, 6), f"({post*100:.2f}%)")

# b)  P(B|+) = 0.5 
spec_min = 1 - sens * prev / (1 - prev)
print("b) Minimum specificity =", round(spec_min, 6), f"({spec_min*100:.2f}%)")
```