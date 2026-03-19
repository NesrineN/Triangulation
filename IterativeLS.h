#ifndef ITERATIVE_LS_H
#define ITERATIVE_LS_H

#include "libOrsa/libNumerics/matrix.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Triangulation {

    /**
     * @brief Performs 3D triangulation using the Iterative LS Method.
     * * @param U Pixel coordinates in the first image (u, v, 1).
     * @param U_prime Pixel coordinates in the second image (u', v', 1).
     * @param P Projection matrix of the first camera (3x4).
     * @param P_prime Projection matrix of the second camera (3x4).
     * @return The reconstructed 3D point, or (0,0,0) if failed/behind camera.
     */
    Vec Triangulate_Iterative_LS(
        const Vec& U, 
        const Vec& U_prime, 
        const Mat& P, 
        const Mat& P_prime
    );

} // namespace Triangulation

#endif // ITERATIVE_LS_H