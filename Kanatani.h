#ifndef KANATANI_H
#define KANATANI_H

#include "libOrsa/libNumerics/matrix.h"
#include <vector>
#include <utility>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace OptCorrection {

    std::pair<Vec, Vec> ComputeCorrectedPairs(const Vec& U, const Vec& U_prime, const Mat& F);

} // namespace OptCorrection

namespace Triangulation {

    Vec Triangulate_Kanatani(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr);

    } // namespace Triangulation

#endif // KANATANI_H