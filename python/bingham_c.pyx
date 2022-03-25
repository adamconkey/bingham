import cython
import numpy as np
import matplotlib.pylab as plab
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

import ctypes
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
        return bingham_c.bingham_pdf(<double*> x.data, &self._c_bingham_t)

    def log_pdf(self, np.ndarray[double, ndim=1, mode="c"] x not None):
        return np.log(self.pdf(x))
    
    @cython.boundscheck(False)
    @cython.wraparound(False)    
    def fit(self, np.ndarray[np.float64_t, ndim=2, mode="c"] X):
        cdef int n = X.shape[0]
        cdef int d = X.shape[1]
        cdef double** c_X = bingham_c.new_matrix2(n, 4)
        try:
            for i in range(n):
                c_X[i] = &X[i,0]
            bingham_c.bingham_fit(&self._c_bingham_t, &c_X[0], n, d)
        finally:
            PyMem_Free(c_X)
        self.compute_stats()

    @cython.boundscheck(False)
    @cython.wraparound(False)        
    def sample(self, int n_samples):
        cdef np.ndarray[double, ndim=2, mode="c"] samples
        samples = np.ascontiguousarray(np.empty((n_samples, 4)), dtype=ctypes.c_double)
        cdef double** c_samples = bingham_c.new_matrix2(n_samples, 4)
        try:
            for i in range(n_samples):
                c_samples[i] = &samples[i,0]
            bingham_c.bingham_sample(&c_samples[0], &self._c_bingham_t, n_samples)
        finally:
            PyMem_Free(c_samples)
        return samples

    def compute_stats(self):
        bingham_c.bingham_stats(&self._c_bingham_t)

    @property
    def entropy(self):
        return self._c_bingham_t.stats.entropy

    @property
    def mode(self):
        cdef np.ndarray[double, ndim=1, mode="c"] mode = np.empty(4)
        for i in range(4):
            mode[i] = self._c_bingham_t.stats.mode[i]
        return mode

    @property
    def V(self):
        cdef np.ndarray[double, ndim=2, mode="c"] V = np.empty((3, 4))
        for i in range(3):
            for j in range(4):
                V[i,j] = self._c_bingham_t.V[i][j]
        return V

    @property
    def Z(self):
        cdef np.ndarray[double, ndim=1, mode='c'] Z = np.empty(3)
        for i in range(3):
            Z[i] = self._c_bingham_t.Z[i]
        return Z
        
    def draw(self, n_samples_axis=100, n_orientation_samples=200, vm_bandwidth=50.):
        qs = self.sample(n_orientation_samples)
        Rs = [Quaternion(q).rotation_matrix for q in qs]
        
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        base_coordinates = Quaternion(self.mode).rotation_matrix.T
        
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        self._plot_coordinate_axes(base_coordinates)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1., 1., 1.))
        ax.set_axis_off()
        
        for R in Rs:            
            x = R[:,0]
            y = R[:,1]
            z = R[:,2]
            ax.scatter(x[0], x[1], x[2], color='red', alpha=0.5)
            ax.scatter(y[0], y[1], y[2], color='green', alpha=0.5)
            ax.scatter(z[0], z[1], z[2], color='blue', alpha=0.5)

        # Draw sphere
        N=40
        stride=1
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, cstride=stride, rstride=stride,
                        alpha=0.1, color='slategrey')
        ax.plot_wireframe(x, y, z, cstride=stride, rstride=stride,
                          color='slategrey', lw=0.2)
        plt.show()

    def _plot_coordinate_axes(self, coordinates):
        zeros = np.zeros(3)        
        x, y, z = zip(zeros, coordinates[0])
        plt.plot(x, y, z, linewidth=3, color='red')
        x, y, z = zip(zeros, coordinates[1])
        plt.plot(x, y, z, linewidth=3, color='green')
        x, y, z = zip(zeros, coordinates[2])
        plt.plot(x, y, z, linewidth=3, color='blue')
    

def bingham_cross_entropy(Bingham B1, Bingham B2):
    cdef bingham_c.bingham_t *c_B1 = &B1._c_bingham_t
    cdef bingham_c.bingham_t *c_B2 = &B2._c_bingham_t
    return bingham_c.bingham_cross_entropy(c_B1, c_B2)


def bingham_kl_divergence(Bingham B1, Bingham B2):
    cdef bingham_c.bingham_t *c_B1 = &B1._c_bingham_t
    cdef bingham_c.bingham_t *c_B2 = &B2._c_bingham_t
    return bingham_c.bingham_KL_divergence(c_B1, c_B2)


def bingham_F(np.ndarray[double, ndim=1, mode="c"] Z not None):
    return bingham_c.bingham_F_lookup_3d(<double*> Z.data)


def bingham_dF(Z):
    cdef double dF1 = bingham_c.bingham_dF1_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    cdef double dF2 = bingham_c.bingham_dF2_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    cdef double dF3 = bingham_c.bingham_dF3_3d(<double> Z[0], <double> Z[1], <double> Z[2])
    return np.asarray([dF1, dF2, dF3])


def bingham_pre_rotate_3d(Bingham B, np.ndarray[double, ndim=1, mode="c"] q not None):
    B_rot = Bingham()
    cdef bingham_c.bingham_t *c_B = &B._c_bingham_t
    cdef bingham_c.bingham_t *c_B_rot = &B_rot._c_bingham_t
    bingham_c.bingham_pre_rotate_3d(c_B_rot, c_B, <double*> q.data)
    B_rot.compute_stats()
    return B_rot
