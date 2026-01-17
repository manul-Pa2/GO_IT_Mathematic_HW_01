import numpy as np

# Вхідні матриці (як у прикладі)
M = np.array([
    [100, 150, 200],
    [ 50, 100, 150],
    [  0,  50, 100]
], dtype=float)

E = np.array([
    [20, 30, 40],
    [10, 20, 30],
    [ 5, 10, 15]
], dtype=float)

# 1) Контраст:
contrast = 0.5 * M

# 2) Яскравість:
brightness = M + 25

# 3) Blending (змішування)
blend = 0.8 * M + 0.2 * E

contrast_expected = np.array([     # Аналітичні результати для перевірки
    [50, 75, 100],
    [25, 50,  75],
    [ 0, 25,  50]
], dtype=float)

brightness_expected = np.array([
    [125, 175, 225],
    [ 75, 125, 175],
    [ 25,  75, 125]
], dtype=float)

blend_expected = np.array([
    [84, 126, 168],
    [42,  84, 126],
    [ 1,  42,  83]
], dtype=float)

# Test
print("OK contrast:", np.allclose(contrast, contrast_expected))
print("OK brightness:", np.allclose(brightness, brightness_expected))
print("OK blend:", np.allclose(blend, blend_expected))

# Результати
print("\n=== Contrast (0.5*M) ===\n", contrast)
print("\n=== Brightness (M+25) ===\n", brightness)
print("\n=== Blending (0.8*M + 0.2*E) ===\n", blend)
