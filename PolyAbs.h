#ifndef POLY_ABS_TRIANGULATION_H
#define POLY_ABS_TRIANGULATION_H

#include "libOrsa/libNumerics/matrix.h"
#include "LinearEigen.h" 
#include "PolyBasis.h"
#include <utility> // For std::pair

// Type Aliases to match your implementation
typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Triangulation {

    /**
     * Performs Triangulation using Absolute Value minimization (Poly-Abs).
     * 1. Computes the Fundamental Matrix F.
     * 2. Corrects image points U and U_prime using an 8th-degree polynomial 
     * to minimize the absolute epipolar distance.
     * 3. Performs Linear Eigen triangulation on the corrected optimal points.
     * * @return A 3D vector representing the reconstructed world point.
     */
    Vec Triangulate_Poly_Abs(const Vec& U, 
                             const Vec& U_prime, 
                             const Mat& P, 
                             const Mat& P_prime, 
                             const Mat& K, 
                             const Mat& Rl, 
                             const Mat& Rr, 
                             const Vec& Tl, 
                             const Vec& Tr);

} // namespace Triangulation

#endif // POLY_ABS_TRIANGULATION_H