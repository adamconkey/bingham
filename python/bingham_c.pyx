import cython
import numpy as np

cimport numpy as np
cimport bingham_c
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef class Bingham:
    cdef bingham_c.bingham_t _c_bingham_t

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, 
                  np.ndarray[double, ndim=1, mode="c"] v1 = None, 
                  np.ndarray[double, ndim=1, mode="c"] v2 = None, 
                  np.ndarray[double, ndim=1, mode="c"] v3 = None, 
                  double z1 = 1, double z2 = 1 , double z3 = 1):
        if(v1 is None):
            bingham_c.bingham_new_uniform(&self._c_bingham_t, 3)
        elif(z2 > 0):
            bingham_c.bingham_new_S3(&self._c_bingham_t, 
                                     <double*> v1.data, <double*> v2.data, <double*> v3.data, 
                                     <double> z1, <double> z1, <double> z1)
        else:
            bingham_c.bingham_new_S3(&self._c_bingham_t, 
                                     <double*> v1.data, <double*> v2.data, <double*> v3.data, 
                                     <double> z1, <double> z2, <double> z3)

    def __dealloc__(self):
        bingham_c.bingham_free(&self._c_bingham_t)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def pdf(self, np.ndarray[double, ndim=1, mode="c"] x not None):
        cdef double f = bingham_c.bingham_pdf(<double*> x.data, &self._c_bingham_t)
        return f

    @cython.boundscheck(False)
    @cython.wraparound(False)    
    def fit(self, np.ndarray[np.float64_t, ndim=2, mode="c"] X):
        cdef int n = X.shape[0]
        cdef int d = X.shape[1]
        cdef double** c_X = <double**>PyMem_Malloc(n * sizeof(double*))
        if not c_X:
            raise MemoryError("Problem allocating memory for data to be fit")
        try:
            for i in range(n):
                c_X[i] = &X[i,0]
            bingham_c.bingham_fit(&self._c_bingham_t, &c_X[0], n, d)
        finally:
            PyMem_Free(c_X)
        self.compute_stats()

    def compute_stats(self):
        bingham_c.bingham_stats(&self._c_bingham_t)

    @property
    def entropy(self):
        cdef double entropy = self._c_bingham_t.stats.entropy
        return entropy
        
        
def bingham_F(np.ndarray[double, ndim=1, mode="c"] Z not None):
    cdef double F = bingham_c.bingham_F_lookup_3d(<double*> Z.data)
    return F


def bingham_dF(Z):
    cdef double dF1 = bingham_c.bingham_dF1_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    cdef double dF2 = bingham_c.bingham_dF2_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    cdef double dF3 = bingham_c.bingham_dF3_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    return np.asarray([dF1, dF2, dF3])


def bingham_cross_entropy(Bingham B1, Bingham B2):
    cdef bingham_c.bingham_t *c_B1 = &B1._c_bingham_t
    cdef bingham_c.bingham_t *c_B2 = &B2._c_bingham_t
    cdef double ce = bingham_c.bingham_cross_entropy(c_B1, c_B2)
    return ce


def bingham_kl_divergence(Bingham B1, Bingham B2):
    cdef bingham_c.bingham_t *c_B1 = &B1._c_bingham_t
    cdef bingham_c.bingham_t *c_B2 = &B2._c_bingham_t
    cdef double kl = bingham_c.bingham_KL_divergence(c_B1, c_B2)
    return kl
