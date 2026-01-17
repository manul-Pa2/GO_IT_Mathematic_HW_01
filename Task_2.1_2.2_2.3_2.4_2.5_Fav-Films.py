import numpy as np

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

# Вектори
u  = np.array([8, 2, 5], dtype=float)
vA = np.array([9, 1, 2], dtype=float)
vB = np.array([1, 9, 8], dtype=float)
vC = np.array([7, 2, 6], dtype=float)

# Обчислення
simA = cosine_similarity(u, vA)
simB = cosine_similarity(u, vB)
simC = cosine_similarity(u, vC)

print("||u|| =", np.linalg.norm(u))

print("cos(u, A) =", simA)
print("cos(u, B) =", simB)
print("cos(u, C) =", simC)

# Визначимо найкращий
sims = {"A": simA, "B": simB, "C": simC}
best_movie = max(sims, key=sims.get)

print("\nНайкраще підходить фільм:", best_movie, "зі схожістю", sims[best_movie])

# Перевірка з аналітичними значеннями
expected = {"A": 0.9392659661, "B": 0.5664036370, "C": 0.9892499383}
print("\nПеревірка allclose:")
for k in sims:
    print(k, np.allclose(sims[k], expected[k], rtol=1e-9, atol=1e-9))

# Найкраще підходить фільм vC (Drama Movie), - мені також більш імпонують драми))
