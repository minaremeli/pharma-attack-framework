# cython: language_level=3

cimport intersection
cimport numpy as np
import numpy as np

cpdef np.ndarray[np.uint32_t, ndim=1] intersect(set1, set2):
    if not set1.flags['C_CONTIGUOUS']:
        set1 = np.ascontiguousarray(set1) # Makes a contiguous copy of the numpy array.
    
    cdef const np.uint32_t[:] s1_view = set1
    
    if not set2.flags['C_CONTIGUOUS']:
        set2 = np.ascontiguousarray(set2) # Makes a contiguous copy of the numpy array.
    
    cdef const np.uint32_t[:] s2_view = set2
    
    result = np.empty(min(set1.size, set2.size), dtype = np.uint32)
    
    if not result.flags['C_CONTIGUOUS']:
        result = np.ascontiguousarray(result) # Makes a contiguous copy of the numpy array.
    
    cdef np.uint32_t[:] r_view = result

    #/*
    #* Given two arrays, this writes the intersection to out. Returns the
    #* cardinality of the intersection.
    #*
    #* This is a mix of very fast vectorized intersection algorithms, several
    #* designed by N. Kurz, with adaptations by D. Lemire.
    #*/
    size_res = intersection.SIMDintersection(&s1_view[0], set1.size, &s2_view[0], set2.size, &r_view[0]);

    return result[:size_res]
