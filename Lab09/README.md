# README (scurt) — Explicatii pentru (b) si (d)

## (b) Efectul lui Y si theta asupra posteriorului lui n

Setare: n ~ Poisson(10), Y | n, theta ~ Binomial(n, theta), cu theta cunoscut.

Idei cheie:
- Y mai mare => posterior(n) se deplaseaza spre valori mai mari (mai multi vizitatori necesari).
- theta mai mare (cu Y fix) => posterior(n) se deplaseaza spre valori mai mici (fiecare vizitator are sanse mai mari sa cumpere).

Comparatii rapide:
- Y=0: cu theta=0.5, zero cumparatori este plauzibil doar pentru n mici; posteriorul favorizeaza n mai mic decat la theta=0.2.
- Y=5: centrul posteriorului pentru n este mai jos la theta=0.5 decat la theta=0.2.
- Y=10: posteriorul muta n in sus in ambele cazuri; totusi, pentru theta=0.5, n necesar este mult mai mic decat pentru theta=0.2.

Rezumat: creste Y -> n creste; creste theta -> n scade (pentru acelasi Y).

---

## (d) Posterior predictiv vs posterior pentru n

- Posterior pentru n: p(n | Y, theta) — credinta asupra numarului latent de vizitatori.
- Posterior predictiv pentru Y*: p(Y* | Y, theta) = sum_n p(Y* | n, theta) p(n | Y, theta), cu p(Y* | n, theta) = Binomial(n, theta).

Diferente:
- Tinta: p(n|data) descrie n (latent); p(Y*|data) descrie o observatie viitoare.
- Dispersie: predictivul include atat variatia binomiala pentru Y*, cat si incertitudinea despre n => de obicei mai lat.
- Interpretare: posterior(n) raspunde "cati vizitatori au fost plauzibil azi?", predictivul raspunde "ce am putea observa data viitoare?".
