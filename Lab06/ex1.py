# Datele problemei
prev = 0.01      # P(B)
sens = 0.95      # P(+|B) 
spec = 0.90      # P(-| not B) 

post = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
print(f"a) P(B | +) = {post:.6f} (~{post*100:.2f}%)")

spec_min = 1 - sens * prev / (1 - prev)
print(f"b) Specificitate minimă = {spec_min:.6f} (~{spec_min*100:.2f}%)")

# Verificare
post_check = (sens * prev) / (sens * prev + (1 - spec_min) * (1 - prev))
print(f"Verificare: P(B | +) la spec_min = {post_check:.6f}")
