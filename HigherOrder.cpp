#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <complex>
#include <vector>
#include <cmath>
#include "PolyBasis.h"
#include "LinearEigen.h"

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

Mat ComputeV0Matrix(double x, double xp, double y, double yp, double f0){

    Mat V0_delta=Mat::zeros(9);
    V0_delta(0,0)=x*x + xp*xp;

    V0_delta(0,1)=xp*yp;
    V0_delta(1,0)=xp*yp;
    V0_delta(3,4)=xp*yp;
    V0_delta(4,3)=xp*yp;

    V0_delta(0,2)=f0*xp;
    V0_delta(2,0)=f0*xp;
    V0_delta(3,5)=f0*xp;
    V0_delta(5,3)=f0*xp;

    V0_delta(0,3)= x*y;
    V0_delta(3,0)= x*y;
    V0_delta(1,4)= x*y;
    V0_delta(4,1)= x*y;

    V0_delta(0,6)=f0*x;
    V0_delta(6,0)=f0*x;
    V0_delta(1,7)=f0*x;
    V0_delta(7,1)=f0*x;

    V0_delta(1,1)=x*x + yp*yp;
    V0_delta(3,3)=y*y + xp*xp;
    V0_delta(4,4)=y*y + yp*yp;
    V0_delta(2,2)=f0*f0;
    V0_delta(5,5)=f0*f0;
    V0_delta(6,6)=f0*f0;
    V0_delta(7,7)=f0*f0;

    V0_delta(1,2)=f0*yp;
    V0_delta(2,1)=f0*yp;
    V0_delta(4,5)=f0*yp;
    V0_delta(5,4)=f0*yp;
    
    V0_delta(3,6)=f0*y;
    V0_delta(6,3)=f0*y;
    V0_delta(4,7)=f0*y;
    V0_delta(7,4)=f0*y;

    return V0_delta;

}

Vec ComputeDeltaHat(double xhat, double xhatp, double xtilde, double xtildep, double yhat, double yhatp, double ytilde, double ytildep, double f0){

    Vec Vhat(9);
    Vhat(0)= xhat*xhatp + xhatp*xtilde + xhat*xtildep;
    Vhat(1)= xhat*yhatp + yhatp*xtilde + xhat*ytildep;
    Vhat(2)= f0*(xhat+xtilde);
    Vhat(3)= yhat*xhatp + xhatp*ytilde + yhat*xtildep;
    Vhat(4)= yhat*yhatp + yhatp*ytilde + yhat*ytildep;
    Vhat(5)= f0* (yhat+ytilde);
    Vhat(6)= f0*(xhatp+xtildep);
    Vhat(7)= f0*(yhatp+ytildep);
    Vhat(8)= f0*f0;

    return Vhat;
}

namespace OptCorrection
{
    std::pair<Vec, Vec> ComputeCorrectedPairs_Higher(const Vec& U, const Vec& U_prime, const Mat& F){

        double x=U(0);
        double y=U(1);
        double xp=U_prime(0);
        double yp=U_prime(1);

        Mat Pk=Mat::eye(3);
        Pk(2,2)=0;

        double f0 = 600;

        Vec u(9);
        u(0)=F(0,0);
        u(1)=F(0,1);
        u(2)=F(0,2);
        u(3)=F(1,0);
        u(4)=F(1,1);
        u(5)=F(1,2);
        u(6)=F(2,0);
        u(7)=F(2,1);
        u(8)=F(2,2);


        double E0=1e15; 
        double xhat=x;
        double yhat=y;
        double xhatp=xp;
        double yhatp=yp;
        double xtilde=0;
        double ytilde=0;
        double xtildep=0;
        double ytildep=0;

        for(int i=0; i<100; i++){
            Vec Delta_Hat=ComputeDeltaHat(xhat, xhatp, xtilde, xtildep, yhat, yhatp, ytilde, ytildep, f0);
            Mat V0_delta_hat=ComputeV0Matrix(xhat, xhatp, yhat, yhatp, f0);

            // updating xtilde and ytilde:
            double denom=dot(u,V0_delta_hat*u);
            if(std::abs(denom) < 1e-12){return std::pair<Vec, Vec>(Vec(0,0,0), Vec(0,0,0));}
            double scalar=dot(u, Delta_Hat)/denom;
            Mat Xtilde= scalar*(F.copy(0,1,0,2)*Vec(xhatp, yhatp, f0));
            xtilde=Xtilde(0,0);
            ytilde=Xtilde(1,0);

            // updating xtildep and ytildep:
            Mat Xtildep= scalar*(F.t().copy(0,1,0,2)*Vec(xhat, yhat, f0));
            xtildep=Xtildep(0,0);
            ytildep=Xtildep(1,0);

            double E=(xtilde*xtilde + ytilde*ytilde + xtildep*xtildep + ytildep*ytildep)/(f0*f0);
            if(std::abs(E-E0)< 1e-9){return std::pair<Vec, Vec>(Vec(xhat, yhat, 1),Vec(xhatp, yhatp, 1));}

            E0=E;
            xhat=x-xtilde;
            yhat=y-ytilde;
            xhatp=xp-xtildep;
            yhatp=yp-ytildep;
        }
        return std::pair<Vec, Vec>(Vec(xhat, yhat, 1),Vec(xhatp, yhatp, 1));
    }

} // namespace OptCorrection

namespace Triangulation {

    Vec Triangulate_HigherOrder(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime, const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr){

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
        auto [U_hat, U_p_hat] = OptCorrection::ComputeCorrectedPairs_Higher(U, U_prime, F); 

        // we then use the corrected pair and P and P_prime to do triangulation using linear Eigen
        Vec result= Triangulation::Triangulate_Linear_Eigen(U_hat, U_p_hat, P, P_prime);
        return result;
    }
    
}