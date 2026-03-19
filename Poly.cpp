#include "libOrsa/libNumerics/matrix.h"
#include "LinearEigen.h"
#include "PolyBasis.h"
#include <iostream>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;


namespace Triangulation {

    Vec Triangulate_Poly(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr){

        // we compute F
        Mat F=Poly::ComputeFundamentalMatrix(K, Rl, Rr, Tl, Tr);

        std::cout << "--- Matrix F Debug ---" << std::endl;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << F(i, j) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << "----------------------" << std::endl;

        // we correct U and U_prime
        auto [U_hat, U_p_hat] = Poly::ComputeCorrectedPairs(U, U_prime, F); 

        // we then use the corrected pair and P and P_prime to do triangulation using linear Eigen
        Vec result= Triangulation::Triangulate_Linear_Eigen(U_hat, U_p_hat, P, P_prime);
        return result;
    }
    
}

