#!/usr/bin/env python
import sys
import pybingham
import numpy as np
from pyquaternion import Quaternion


def get_random_quats(n_samples=100, dist_threshold=0.3, origin=[1,0,0,0]):
    origin = Quaternion(origin)
    X = []
    while len(X) < n_samples:
        q = Quaternion.random()
        dist = Quaternion.distance(origin, q)
        if dist < dist_threshold:
            X.append(q.elements)
    return np.stack(X)


print("\nGetting random quaternions to fit distributions with...")
X1 = get_random_quats()
X2 = get_random_quats()
print("Sample generation complete")
        
b1 = pybingham.Bingham()
b1.fit(X1)
b2 = pybingham.Bingham()
b2.fit(X2)

print("B1 PDF", b1.pdf(np.array([1,0,0,0], dtype=np.double)))
print("B2 PDF", b2.pdf(np.array([1,0,0,0], dtype=np.double)))
print("B1 ENTROPY", b1.entropy)
print("B1 MODE", b1.mode)
print("B1 V", b1.V)
V = np.concatenate([b1.V, np.expand_dims(b1.mode, 0)])
print("B1 V orthogonal?", np.allclose(V.T @ V, np.eye(4), atol=1e-10))
print("B1 Z", b1.Z)
print("B1 3 SAMPLES:", b1.sample(3))

print("CE", pybingham.bingham_cross_entropy(b1, b2))
print("KL", pybingham.bingham_kl_divergence(b1, b2))

# print("Drawing...")
# b2.draw()

print("DONE")
