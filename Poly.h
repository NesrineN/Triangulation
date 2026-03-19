#ifndef POLY_TRIANGULATION_H
#define POLY_TRIANGULATION_H

#include "libOrsa/libNumerics/matrix.h"
#include "LinearEigen.h" // For Triangulate_Linear_Eigen
#include "PolyBasis.h"  // For Poly namespace functions

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Triangulation {

    /**
     * Performs Optimal Triangulation (Hartley-Sturm).
     * 1. Computes the Fundamental Matrix F.
     * 2. Corrects the image points U and U_prime to satisfy the epipolar constraint.
     * 3. Uses the corrected points to find the 3D coordinate via Linear Eigen triangulation.
     * * @return A 3D vector (X, Y, Z) representing the reconstructed world point.
     */
    Vec Triangulate_Poly(const Vec& U, 
                         const Vec& U_prime, 
                         const Mat& P, 
                         const Mat& P_prime, 
                         const Mat& K, 
                         const Mat& Rl, 
                         const Mat& Rr, 
                         const Vec& Tl, 
                         const Vec& Tr);

} // namespace Triangulation

#endif // POLY_TRIANGULATION_H