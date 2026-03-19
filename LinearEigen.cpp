#include "LinearEigen.h"
#include "libOrsa/libNumerics/matrix.h"
#include <iostream>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;


namespace Triangulation {

// function that performs the triangulation using the Linear Eigen Method: 
// takes 2 pixel points u and u' and P and P' the projection matrices and returns the 3D point X upon triangulation
// u=PX and u'=P'X
// function must create Matrix A which has the 4 equations and solve AX=0 using SVD decomposition
// rows of matrix A:
// uP3T-P1T
// vP3T-P2T
// u'P3'T - P1'T
// v'P3'T - P2'T

// U and U_prime are in the form of u=w(u,v,1)
Vec Triangulate_Linear_Eigen(const Vec& U, const Vec& U_prime, const Mat& P, const Mat& P_prime){
    // extracting the coordinates from the u and u' vectors
    double u= U(0);
    double v= U(1);
    double u_p= U_prime(0);
    double v_p= U_prime(1);

    // creating the matrix A that is 4x4
    Mat A= Mat::zeros(4,4);

    Mat p0=P.copyRows(0,0);
    Mat p1=P.copyRows(1,1);
    Mat p2=P.copyRows(2, 2);

    Mat p0p=P_prime.copyRows(0,0);
    Mat p1p=P_prime.copyRows(1,1);
    Mat p2p=P_prime.copyRows(2, 2);

    // Row 0: uP3T-P1T
    Mat row0= u*p2 - p0;
    // Row 1: vP3T-P2T
    Mat row1= v*p2 - p1;
    // Row 2: u'P3'T - P1'T  
    Mat row2= u_p*p2p - p0p;
    // Row 3: v'P3'T - P2'T
    Mat row3= v_p*p2p - p1p;

    A.paste(0, 0, row0);
    A.paste(1,0, row1);
    A.paste(2,0, row2);
    A.paste(3,0, row3);


    // std::cout << "--- Matrix A Debug ---" << std::endl;
    // for (int i = 0; i < 4; ++i) {
    //     for (int j = 0; j < 4; ++j) {
    //         std::cout << A(i, j) << "\t";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "----------------------" << std::endl;

    // Solving Minimum of AX subject to ||X||=1 using SVD Decomposition
    Mat W(4,4);
    Mat V(4,4);
    Vec S(4);
    A.SVD(W,S,V);

    int minIndex = 0;
    for (int i = 1; i < 4; ++i)
        if (S(i) < S(minIndex))
            minIndex = i;

    Vec solution = V.col(minIndex);

    double w_hom = solution(3);

    if(std::abs(w_hom) > 1e-9){
        double X= solution(0) / w_hom;
        double Y= solution(1) / w_hom;
        double Z= solution(2) / w_hom;

        if(Z<0){
            std::cout << "Z was negative" << std::endl;
            return Vec(0,0,0);// point is behind the camera
        }

        return Vec(X,Y,Z);
    }
    else{
        std::cout << "w_hom was zero" << std::endl;
        return Vec(0,0,0); // point at inifinity
    }
}

}


