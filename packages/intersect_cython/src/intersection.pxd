# extern blocks define interfaces for Cython to C code
from libc.stdint cimport uint32_t

cdef extern from "intersection.h":
    #double run_dp(double sigma, double* c, int tau, dgs_disc_gauss_alg_t alg, long* samples_out, size_t ntrials, unsigned int seed)
    
    #/*
    #* Given two arrays, this writes the intersection to out. Returns the
    #* cardinality of the intersection.
    #*
    #* This is a mix of very fast vectorized intersection algorithms, several
    #* designed by N. Kurz, with adaptations by D. Lemire.
    #*/
    size_t SIMDintersection(const uint32_t *set1, const size_t length1, const uint32_t *set2, const size_t length2, uint32_t *out);

