import numpy as np

# Данні
t = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([22, 28, 37, 45, 53], dtype=float)

# 1) Матриця A: [t, 1]
A = np.column_stack([t, np.ones_like(t)])

# 2) МНК через lstsq: A * [k, b] = y
(k, b), residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

# 3) Прогноз на 6-ту годину
t_pred = 6
y_pred = k * t_pred + b

print("=== np.linalg.lstsq ===")
print(f"k = {k:.4f}, b = {b:.4f}")
print(f"Тренд: y = {k:.4f} * t + {b:.4f}")
print(f"Прогноз на t=6: ŷ = {y_pred:.4f}")

# ===== Нормальне рівняння =======
ATA = A.T * A
ATy = A.T * y
x = np.linalg.solve(ATA, ATy)      # x = [k, b]

k2, b2 = x
y_pred2 = k2 * t_pred + b2

print("\n=== Нормальне рівняння (AᵀA x = Aᵀy) ===")
print("A^T A =\n", ATA)
print("A^T y =", ATy)
print(f"k = {k2:.4f}, b = {b2:.4f}")
print(f"Тренд: y = {k2:.4f} * t + {b2:.4f}")
print(f"Прогноз на t=6: ŷ = {y_pred2:.4f}")

print("\nРізниця між методами:")
print(f"|Δk| = {abs(k-k2):.10f}, |Δb| = {abs(b-b2):.10f}")

# Test:   k = 7.9   b = 13.3    
# Рівняння тренду: y = 7.9·t + 13.3
# Прогноз на 6-ту годину: ŷ = 60.7     ("ŷ"-взяв з умови, сподіваюсь синтаксис не конфліктує)
