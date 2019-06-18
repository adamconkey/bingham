#from bingham_c cimport bingham_t as c_bingham_t
#from bingham_c cimport bingham_new_uniform as c_bingham_new_uniform

import cython

cimport numpy as np
import numpy as np

cimport bingham_c

cdef class Bingham:
    cdef bingham_c.bingham_t _c_bingham_t

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, 
                  np.ndarray[double, ndim=1, mode="c"] v1 = None, 
                  np.ndarray[double, ndim=1, mode="c"] v2 = None, 
                  np.ndarray[double, ndim=1, mode="c"] v3 = None, 
                  double z1 = 1, double z2 = 1 , double z3 = 1,
                  #bingham_c.bingham_t c_bingham_t = None,
                  ):
        #if(c_bingham_t is not None):
        #    self._c_bingham_t = c_bingham_t
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

def bingham_F(np.ndarray[double, ndim=1, mode="c"] Z not None):
    cdef double F = bingham_c.bingham_F_lookup_3d(<double*> Z.data)
    return F

def bingham_dF(Z):
    cdef double dF1 = bingham_c.bingham_dF1_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    cdef double dF2 = bingham_c.bingham_dF2_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    cdef double dF3 = bingham_c.bingham_dF3_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    return np.asarray([dF1, dF2, dF3])

def bingham_mult(Bingham B, Bingham B1, Bingham B2):
    #cdef bingham_c.bingham_t *b = &B._c_bingham_t
    #cdef bingham_c.bingham_t *b1 = &B1._c_bingham_t
    #cdef bingham_c.bingham_t *b2 = &B2._c_bingham_t
    #bingham_c.bingham_mult(b, b1, b2)
    bingham_c.bingham_mult(<bingham_c.bingham_t *> &B._c_bingham_t, 
                           <bingham_c.bingham_t *> &B1._c_bingham_t,
                           <bingham_c.bingham_t *> &B2._c_bingham_t)
    #b = NULL
    #b1 = NULL
    #b2 = NULL
    #return None

#def bingham_multi_array(B_out, B_array):


