#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <complex>
#include <vector>
#include <cmath>
#include "PolyBasis.h"
#include "LinearEigen.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace OptCorrection
{
    std::pair<Vec, Vec> ComputeCorrectedPairs(const Vec& U, const Vec& U_prime, const Mat& F){
        
        Vec x(3); 
        x(0) = U(0); 
        x(1) = U(1); 
        x(2) = 1.0;

        Vec x_prime(3); 
        x_prime(0) = U_prime(0); 
        x_prime(1) = U_prime(1); 
        x_prime(2) = 1.0;


        // first we form the diag(1,1,0) matrix Pk 
        Mat Pk=Mat::eye(3);
        Pk(2,2)=0;

        // then, we compute the corrections delta x and delta x': 

        // // we get the inner product (x, Fx')
        Vec Fxp=F*x_prime; 
        Vec PkFxp=Pk*Fxp;
        Vec FTx=F.t()*x;
        Vec PkFTx=Pk*FTx;
        double xFxp= dot(x,Fxp);

        // preparing the denominator of delta x and delta x':
        // (Fx',PkFx')
        double FxpPkFxp=dot(Fxp, PkFxp);
        double FTxPkFTx= dot(FTx, PkFTx);
        double denom= FxpPkFxp + FTxPkFTx;
        if(std::abs(denom) < 1e-12){return std::pair<Vec, Vec>(Vec(0,0,0), Vec(0,0,0));}

        Vec nom= xFxp*(PkFxp);

        Vec deltax=nom/denom;

        Vec nomp=xFxp*(PkFTx);

        Vec deltaxp=nomp/denom;

        // we apply the corrections and return the corrected points
        Vec xhat=x-deltax;
        Vec xhatp=x_prime-deltaxp;

        // should we normalize here before returning? i.e. should we divide xhat(0) and xhat(1) by xhat(2) and set xhat(2) to 1? (same thing for xhatp)

        return std::pair<Vec, Vec>(xhat, xhatp);

    }

} // namespace OptCorrection

namespace Triangulation {

    Vec Triangulate_Kanatani(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr){

        // we compute F
        Mat F=Poly::ComputeFundamentalMatrix(K, Rl, Rr, Tl, Tr);

        // std::cout << "--- Matrix F Debug ---" << std::endl;
        // for (int i = 0; i < 3; ++i) {
        //     for (int j = 0; j < 3; ++j) {
        //         std::cout << F(i, j) << "\t";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "---------------- ------" << std::endl;

        // we correct U and U_prime
        auto [U_hat, U_p_hat] = OptCorrection::ComputeCorrectedPairs(U, U_prime, F); 

        // we then use the corrected pair and P and P_prime to do triangulation using linear Eigen
        Vec result= Triangulation::Triangulate_Linear_Eigen(U_hat, U_p_hat, P, P_prime);
        return result;
    }
    
}


// normalized vs non-normalized points? if i use F, i have to use non-normalized pixel coordinates? what does that mean