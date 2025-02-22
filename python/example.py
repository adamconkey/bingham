import pybingham
import numpy as np


v1 = np.array([0,1,0,0], dtype=np.double)
v2 = np.array([0,0,1,0], dtype=np.double)
v3 = np.array([0,0,0,1], dtype=np.double)
B1 = pybingham.Bingham(v1, v2, v3, -1, -2, -3)
B2 = pybingham.Bingham(v1, v2, v3, -5, -7, -3)
Bm = pybingham.Bingham()
pybingham.bingham_mult(Bm, B1, B2)

print(f"PDF = {Bm.pdf(np.array([1,0,0,0], dtype=np.double))}")
