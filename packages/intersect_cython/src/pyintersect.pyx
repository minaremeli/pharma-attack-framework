# cython: language_level=3

cimport intersection
cimport numpy as cnp
import numpy as np


cpdef bint match(nnz_indices, grad_nnz, int hidden_size, float voting_threshold):
    if not nnz_indices.flags['C_CONTIGUOUS']:
        nnz_indices = np.ascontiguousarray(nnz_indices) # Makes a contiguous copy of the numpy array.
    
    cdef const int[::1] nnz_indices_view = nnz_indices
    
    if not grad_nnz.flags['C_CONTIGUOUS']:
        grad_nnz = np.ascontiguousarray(grad_nnz) # Makes a contiguous copy of the numpy array.
    
    cdef const int[::1] grad_nnz_view = grad_nnz
    
    cdef bint ret = intersection.match(&nnz_indices_view[0], nnz_indices.size, &grad_nnz_view[0], grad_nnz.size, hidden_size, voting_threshold)

    return ret

cpdef cnp.ndarray[cnp.uint32_t, ndim=1] intersect(set1, set2):
    if not set1.flags['C_CONTIGUOUS']:
        set1 = np.ascontiguousarray(set1) # Makes a contiguous copy of the numpy array.
    
    cdef const cnp.uint32_t[::1] s1_view = set1
    
    if not set2.flags['C_CONTIGUOUS']:
        set2 = np.ascontiguousarray(set2) # Makes a contiguous copy of the numpy array.
    
    cdef const cnp.uint32_t[::1] s2_view = set2
    
    result = np.empty(min(set1.size, set2.size), dtype = np.uint32)
    
    if not result.flags['C_CONTIGUOUS']:
        result = np.ascontiguousarray(result) # Makes a contiguous copy of the numpy array.
    
    cdef cnp.uint32_t[::1] r_view = result

    #/*
    #* Given two arrays, this writes the intersection to out. Returns the
    #* cardinality of the intersection.
    #*
    #* This is a mix of very fast vectorized intersection algorithms, several
    #* designed by N. Kurz, with adaptations by D. Lemire.
    #*/
    size_res = intersection.SIMDintersection(&s1_view[0], set1.size, &s2_view[0], set2.size, &r_view[0]);

    return result[:size_res]
