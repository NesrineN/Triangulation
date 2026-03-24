#ifndef HIGHERORDER_H
#define HIGHERORDER_H

#include "libOrsa/libNumerics/matrix.h"
#include <vector>
#include <utility>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;


Mat ComputeV0Matrix(double x, double xp, double y, double yp, double f0);
Vec ComputeDeltaHat(double xhat, double xhatp, double xtilde, double xtildep, double yhat, double yhatp, double ytilde, double ytildep, double f0);


namespace OptCorrection {

    std::pair<Vec, Vec> ComputeCorrectedPairs_Higher(const Vec& U, const Vec& U_prime, const Mat& F);

} // namespace OptCorrection

namespace Triangulation {

    Vec Triangulate_HigherOrder(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr);

    } // namespace Triangulation

#endif // HIGHERORDER_H