#include <gsl/gsl_poly.h> // installed on MSYS2 MinGW 64-bit
#include "libOrsa/libNumerics/matrix.h"
#include <iostream> 
#include <complex>
#include <vector>
#include <cmath>
#include <limits>

typedef libNumerics::matrix<double> Mat;
typedef libNumerics::vector<double> Vec;

namespace Poly
{
    // Functions required for computing the corrected pairs

    // function to do the rigid transformations --> TranslateToOrigin placing u=u'=(0,0,1) and place e1 and e2 on the x-axis such that:
    // e1=(1,0,f) and e2=(1,0,f')

    Mat TranslationMatrixToOrigin(const Vec& U){
        Mat result = Mat::eye(3);
        result(0, 2) = -U(0);
        result(1, 2) = -U(1);
        return result;
    } // returns translation matrix denoted by L in the paper . L*u gives the origin 

    Mat RotationMatrixToX(const Vec& e)
    {
        Mat result = Mat::eye(3);
        
        // we calculate the magnitude of the 2D part of the epipole
        double norm = std::sqrt(e(0) * e(0) + e(1) * e(1));
        
        if (norm < 1e-9) {
            return result; // This means the epipole is already at the origin or invalid
        }

        double cos = e(0) / norm;
        double sin = e(1) / norm;

        // To satisfy sin(e1) + cos(e2) = 0:
        // We want to rotate the vector (ex, ey) to (norm, 0)
        result(0,0) = cos;
        result(0,1) = sin;
        result(1,0) = -sin;
        result(1,1) = cos;

        return result;
    }

    // function to compute the Fundamental Matrix F from the Projection Matrices P and P'
    Mat ComputeFundamentalMatrix(const Mat& K, const Mat& Rl, const Mat& Rr, const Vec& Tl, const Vec& Tr){
        Mat R=Rl*Rr.t();
        Vec T= Tl - (R*Tr);
        Mat Tx=Mat::zeros(3);
        Tx(0,1)=-T(2);
        Tx(0,2)=T(1);
        Tx(1,0)=T(2);
        Tx(1,2)=-T(0);
        Tx(2,0)=-T(1);
        Tx(2,1)=T(0);

        Mat E=Tx*R;
        Mat F=((K.inv()).t() * E)*(K.inv());

        return F;
    }

    // function to Compute the Fundamental Matrix after rigid transformations
    Mat TransformFundamentalMatrix(const Mat& F, const Mat& R, const Mat& L, const Mat& R_p, const Mat L_p){
        Mat T = R*L;
        Mat T_p = R_p*L_p;

        Mat F_transformed= T_p.inv().t()*F*T.inv();

        return F_transformed;
    }

    // function to ComputeRightEpipole and ComputeLeftEpipole
    // right epipole: we solve Fe=0 --> SVD
    // left epipole: we solve e^T F=0 --> F^T e=0 --> SVD
    Vec ComputeRightEpipole(const Mat& F){
        Mat W(3,3);
        Mat V(3,3);
        Vec S(3);
        F.SVD(W,S,V);

        int minIndex = 0;
        for (int i = 1; i < 3; ++i)
            if (S(i) < S(minIndex))
                minIndex = i;

        Vec e = V.col(minIndex);

        if (std::abs(e(2)) > 1e-9) {
            e(0) /= e(2);
            e(1) /= e(2);
            e(2) = 1.0;
        }

        return e;
    }

    Vec ComputeLeftEpipole(const Mat& F){
        Mat W(3,3);
        Mat V(3,3);
        Vec S(3);
        F.SVD(W,S,V);

        int minIndex = 0;
        for (int i = 1; i < 3; ++i)
            if (S(i) < S(minIndex))
                minIndex = i;

        Vec e = W.col(minIndex);

        if (std::abs(e(2)) > 1e-9) {
            e(0) /= e(2);
            e(1) /= e(2);
            e(2) = 1.0;
        }

        return e;
    }

    // function to solve the 6-degree polynomial 
    std::vector<double> SolvePoly(const double a, const double b, const double c, const double d, const double f, const double f_p){

        double K = a*d - b*c;
        double f2 = f*f;
        double f4 = f2*f2;
        double f_prime2 = f_p * f_p;

        // Coefficients for L(t) = alpha*t^2 + beta*t + gamma
        double alpha = a*a + f_prime2*c*c;
        double beta  = 2*(a*b + f_prime2*c*d);
        double gamma = b*b + f_prime2*d*d;

        double c6 = -f4 * K * a * c;
        double c5 = alpha * alpha - f4 * K * (a*d + b*c);
        double c4 = 4 * (a*b + f_prime2*c*d) * alpha - f4 * K * b * d - 2 * f2 * K * a * c;
        double c3 = 2 * alpha * gamma + beta * beta - 2 * f2 * K * (a * d + b * c);
        double c2 = 2 * beta * gamma - 2 * f2 * K * b * d - K * a * c;
        double c1 = gamma * gamma - K * (a * d + b * c);
        double c0 = -K * b * d;

        double coeffs[7] = { c0, c1, c2, c3, c4, c5, c6 };

        int actual_degree = 6;
        while (actual_degree > 0 && std::abs(coeffs[actual_degree]) < 1e-15) {
            actual_degree--;
        }

        std::vector<double> z(12);
        gsl_poly_complex_workspace * w = gsl_poly_complex_workspace_alloc (actual_degree+1);
        gsl_poly_complex_solve (coeffs, actual_degree+1, w, z.data());
        gsl_poly_complex_workspace_free (w);

        return z;
    }

    // this function is to solve the 8 degree polynomial obtained in the Poly-Abs method where we minimize the absolute value of the distance from u and u' to lambda and lambda prime instead of the squared distance
    std::vector<double> SolvePolyAbs(const double a, const double b, const double c, const double d, const double f, const double f_p){


        double alpha=a*a+f_p*f_p*c*c;
        double beta=2*(a*b+f_p*f_p*c*d);
        double gamma= (b*b)+(f_p*f_p*d*d);
        
        double coeff8=(alpha*alpha*alpha)-((a*d)-(b*c))*((a*d)-(b*c))*a*a*std::pow(f,6);
        double coeff7=3*alpha*alpha*beta-2*a*b*((a*d)-(b*c))*((a*d)-(b*c))*std::pow(f,6);
        double coeff6=3*alpha*alpha*gamma+ 3*alpha*beta*beta-((a*d)-(b*c))*((a*d)-(b*c))*(a*a + 3*a*a*f*f+ b*b*std::pow(f,6));
        double coeff5=6*alpha*beta*gamma+beta*beta*beta-((a*d)-(b*c))*((a*d)-(b*c))*(2*a*b+6*a*b*f*f);
        double coeff4=3*alpha*gamma*gamma+3*beta*beta*gamma-((a*d)-(b*c))*((a*d)-(b*c))*(b*b+3*b*b*f*f+3*a*a*std::pow(f,4));
        double coeff3=3*beta*gamma*gamma-((a*d)-(b*c))*((a*d)-(b*c))*(6*a*b*f*f+6*a*b*std::pow(f,4));
        double coeff2=3*gamma*gamma*alpha-((a*d)-(b*c))*((a*d)-(b*c))*(3*b*b*f*f+3*a*a*std::pow(f,4));
        double coeff1=3*gamma*gamma*beta-((a*d)-(b*c))*((a*d)-(b*c))*(2*a*b*f*f+2*a*b*std::pow(f,4));
        double coeff0=gamma*gamma*gamma-((a*d)-(b*c))*((a*d)-(b*c))*b*b;


        // double a2=a*a;
        // double a3=a*a*a;
        // double a4=a2*a2;
        // double a5=a4*a;
        // double a6=a3*a3;

        // double b2 = b*b;
        // double b3= b*b*b;
        // double b4= b*b*b*b;
        // double b5= b4*b;
        // double b6=b5*b;

        // double c2=c*c;
        // double c3=c2*c;
        // double c4=c2*c2;
        // double c5=c4*c;
        // double c6=c4*c2;
        
        // double fp2=f_p*f_p;
        // double fp3=fp2*f_p;
        // double fp4= fp2*fp2;
        // double fp5=fp3*fp2;
        // double fp6=fp4*fp2;

        // double d2=d*d;
        // double d3=d2*d;
        // double d4= d2*d2;
        // double d5=d4*d;
        // double d6=d4*d*d;

        // double f2=f*f;
        // double f3=f2*f;
        // double f4=f2*f2;
        // double f5=f4*f;
        // double f6=f3*f3;

        // double coeff8 = a2*f6*(-a2*d2 + 2*a*b*c*d - b2*c2);
        // double coeff7 = 2*a*b*f6*(-a2*d2 + 2*a*b*c*d - b2*c2);
        // double coeff6 = a6 + (3*a4*c2*fp2) - (3*a4*d2*f4) + (6*a3*b*c*d*f4) - (3*a2*b2*c2*f4) - (a2*b2*d2*f6) + (3*a2*c4*fp4) + (2*a*b3*c*d*f6) - (b4*c2*f6) + (c6*fp6);
        // double coeff5 = 6*a5*b + 6*a4*c*d*fp2 + 12*a3*b*c2*fp2 - 6*a3*b*d2*f4 + 12*a2*b2*c*d*f4 + 12*a2*c3*d*fp4 - 6*a*b3*c2*f4 + 6*a*b*c4*fp4 + 6*c5*d*fp6;
        // double coeff4 = 15*a4*b2 - 3*a4*d2*f2 + 3*a4*d2*fp2 + 6*a3*b*c*d*f2 + 24*a3*b*c*d*fp2 - 3*a2*b2*c2*f2 + 18*a2*b2*c2*fp2 - 3*a2*b2*d2*f4 + 18*a2*c2*d2*fp4 + 6*a*b3*c*d*f4 + 24*a*b*c3*d*fp4 - 3*b4*c2*f4 + 3*b2*c4*fp4 + 15*c4*d2*fp6 ;
        // double coeff3 = 20*a3*b3 - 6*a3*b*d2*f2 + 12*a3*b*d2*fp2 + 12*a2*b2*c*d*f2 + 36*a2*b2*c*d*fp2 + 12*a2*c*d3*fp4 - 6*a*b3*c2*f2 + 12*a*b3*c2*fp2 + 36*a*b*c2*d2*fp4 + 12*b2*c3*d*fp4 + 20*c3*d3*fp6 ;
        // double coeff2 = -a4*d2 + 2*a3*b*c*d + 15*a2*b4 - a2*b2*c2 - 3*a2*b2*d2*f2 + 18*a2*b2*d2*fp2 + 3*a2*d4*fp4 + 6*a*b3*c*d*f2 + 24*a*b3*c*d*fp2 + 24*a*b*c*d3*fp4 - 3*b4*c2*f2 + 3*b4*c2*fp2 + 18*b2*c2*d2*fp4 + 15*c2*d4*fp6;
        // double coeff1 = -2*a3*b*d2 + 4*a2*b2*c*d + 6*a*b5 - 2*a*b3*c2 + 12*a*b3*d2*fp2 + 6*a*b*d4*fp4 + 6*b4*c*d*fp2 + 12*b2*c*d3*fp4 + 6*c*d5*fp6;
        // double coeff0 = -a2*b2*d2 + 2*a*b3*c*d + b6 - b4*c2 + 3*b4*d2*fp2 + 3*b2*d4*fp4 + d6*fp6;

        double coeffs[9] = { coeff0, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7, coeff8 };

        int actual_degree = 8;
        while (actual_degree > 0 && std::abs(coeffs[actual_degree]) < 1e-15) {
            actual_degree--;
        }

        std::vector<double> z(16);
        gsl_poly_complex_workspace * w = gsl_poly_complex_workspace_alloc (actual_degree+1);
        gsl_poly_complex_solve (coeffs, actual_degree+1, w, z.data());
        gsl_poly_complex_workspace_free (w);

        return z;
    }

    // function that evaluates the equation s(t) given a t parameter
    double EvaluateEquation(const double& t, const double a, const double b, const double c, const double d, const double f, const double f_p){
        double term1= (t*t) / (1+((t*f)*(t*f)));

        double term2_num= ((c*t)+d)*((c*t)+d);
        double term2_denom=((a*t)+b)*((a*t)+b) + (f_p * f_p)*term2_num;
        if (std::abs(term2_denom) < 1e-20) return term1;

        return (term1+(term2_num/term2_denom));
    }

    double EvaluateEquationAbs(const double& t, const double a, const double b, const double c, const double d, const double f, const double f_p){
        
        double term1= std::abs(t) / std::sqrt((1+((t*f)*(t*f))));

        double term2_num= std::abs(((c*t)+d));
        double term2_denom=((a*t)+b)*((a*t)+b) + (f_p * f_p)*((c*t)+d)*((c*t)+d);
        term2_denom=std::sqrt(term2_denom);

        if (std::abs(term2_denom) < 1e-20) return term1;

        return (term1+(term2_num/term2_denom));
    }

    // function that decides which root gives the minimum s(t)
    // we only want the real roots
    double FindBestRoot(const std::vector<double>& roots, const double a, const double b, const double c, const double d, const double f, const double f_p){
        double minimum = std::numeric_limits<double>::infinity();
        double st_inf = (1.0 / (f*f)) + ((c*c) / (a*a + f_p*f_p*c*c));
        double best_root = 0.0;
        bool found = false;

        int degree = roots.size() / 2;
        
        for(int i = 0; i < degree; i++){
            double realPart = roots[2*i];
            double imagPart = roots[2*i + 1];
            
            if (std::abs(imagPart) > 1e-8)
                continue;

            double st=EvaluateEquation(realPart, a, b, c, d, f, f_p);
            if(st<minimum){
                minimum=st;
                best_root=realPart;
                found=true;
            }
            
        }

        if (!found || st_inf < minimum) {
            return std::numeric_limits<double>::infinity();
        }

        return best_root;
    }

    double FindBestRootAbs(const std::vector<double>& roots, const double a, const double b, const double c, const double d, const double f, const double f_p){
        double minimum = std::numeric_limits<double>::infinity();
        double st_inf = (1.0 / std::abs(f)) + (std::abs(c) / std::sqrt(a*a + f_p*f_p*c*c));
        double best_root = 0.0;
        bool found = false;

        int degree = roots.size() / 2;
        
        for(int i = 0; i < degree; i++){
            double realPart = roots[2*i];
            double imagPart = roots[2*i + 1];

            if (std::abs(imagPart) > 1e-8)
                continue;

            double st=EvaluateEquationAbs(realPart, a, b, c, d, f, f_p);
            if(st<minimum){
                minimum=st;
                best_root=realPart;
                found=true;
            }
        }

        if (!found || st_inf < minimum) {
            return std::numeric_limits<double>::infinity();
        }

        return best_root;
    }

    // function that Constructs the epipolar lines lambda(t) and lambda'(t) from the root t that we found and the fundamental matrix transformed F'
    // lambda(t)=(tf, 1, -t)
    // lambda'(t)=(-f'(ct+d), (at+b), (ct+d))
    Vec ComputeLeftEpipolarLine(const double& best_root, const double& f){
        return Vec((best_root*f), 1, (-best_root));
    }

    Vec ComputeRightEpipolarLine(const double& best_root, const double& a, const double& b, const double& c, const double& d, const double& f_p ){
        double x=((-f_p)*((c*best_root)+d));
        double y=(a*best_root)+b;
        double z=(c*best_root)+d;

        return Vec(x,y,z);
    }

    // function to Find the Point On the epipolar Line that is Closest To Origin u or u' --> that point is u^ or u^'
    // Vec FindClosestPointToOrigin(const Vec& lambda){
    //     // since the points U=U'=(0,0,1) are at the origin, the equation:
    //     // u^ = u - (normal)(distance from u to lambda) becomes very simple

    //     double distance=lambda(2)/((lambda(0)*lambda(0)) + (lambda(1)*lambda(1)));
    //     double x=-(lambda(0)*distance);
    //     double y=-(lambda(1)*distance);

    //     return Vec(x,y);
    // }

    Vec FindClosestPointToOrigin(const Vec& lambda){
        double a = lambda(0);
        double b = lambda(1);
        double c = lambda(2);

        double denom = a*a + b*b;

        if (std::abs(denom) < 1e-20) {
            return Vec(0.0, 0.0); // degenerate line
        }

        double scale = -c / denom;

        double x = scale * a;
        double y = scale * b;

        return Vec(x, y);
    }

    // function to reverse the rigid transformations on u^ or u^' and returns u^ or u^' where the rigid transformations are undone
    Vec BackTransform(const Mat& R, const Mat& L, const Vec& U_hat) {
        // Total forward transformation M = R * L
        Mat M = R * L;

        // Total backward transformation (Inverse)
        Mat M_inv = M.inv();

        Vec u_hom(3);
        u_hom(0) = U_hat(0);
        u_hom(1) = U_hat(1);
        u_hom(2) = 1.0; 

        // Transforming back: Matrix * Vector = Vector
        Vec pixel_hom = M_inv * u_hom;

        // De-homogenizing
        double w = pixel_hom(2);
        
        // Safety check for points at infinity
        if(std::abs(w) < 1e-20) {
            return Vec(0.0, 0.0); 
        }

        // Returning the coordinates in the original image plane
        Vec result(2);
        result(0) = pixel_hom(0) / w;
        result(1) = pixel_hom(1) / w;
        
        return result;
    }

    // function that takes the pairs u and u' and their fundamental matrix F and returns the corrected pairs u^ and u^'
    std::pair<Vec, Vec> ComputeCorrectedPairs(const Vec& U, const Vec& U_prime, const Mat& F){

        // std::cout << "--- Matrix F Debug ---" << std::endl;
        // for (int i = 0; i < 3; ++i) {
        //     for (int j = 0; j < 3; ++j) {
        //         std::cout << F(i, j) << "\t";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "----------------------" << std::endl;
        
        // first we apply the rigid transformations to get F_transformed
        Mat L=TranslationMatrixToOrigin(U);
        Mat L_p=TranslationMatrixToOrigin(U_prime);

        Vec LeftEpipole=ComputeLeftEpipole(F); // homogeneous coords
        Vec RightEpipole=ComputeRightEpipole(F);

        Mat R=RotationMatrixToX(LeftEpipole);
        Mat R_p=RotationMatrixToX(RightEpipole);

        Mat F_transformed=TransformFundamentalMatrix(F, R, L, R_p, L_p);

        // std::cout << "--- Matrix F' Debug ---" << std::endl;
        // for (int i = 0; i < 3; ++i) {
        //     for (int j = 0; j < 3; ++j) {
        //         std::cout << F_transformed(i, j) << "\t";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << "----------------------" << std::endl;

        //F_transformed = 
        // ff'd -f'c -f'd
        // -fb    a    b
        // -fd    c    d 

        // then, we get the coefficients of F_transformed a,b,c,d,f,f_p
        double a=F_transformed(1,1);
        double b=F_transformed(1,2);
        double c=F_transformed(2,1);
        double d=F_transformed(2,2);

        if (std::abs(b) < 1e-9 || std::abs(c) < 1e-9) {
            // degenerate case → fallback
            return std::pair<Vec, Vec>(Vec(0,0), Vec(0,0));
        }

        double f=-(F_transformed(1,0)/b);
        double f_p=-(F_transformed(0,1)/c);

        // now, we find the best root that would minimize our cost function
        std::vector<double> roots=SolvePoly(a, b, c, d, f, f_p);
        double best_root=FindBestRoot(roots, a, b, c, d, f, f_p);
        if (std::isinf(best_root)) {
            Vec lambda(f, 0, -1);
            Vec lambda_p(-f_p * c, a, c);

            Vec U_hat_old = FindClosestPointToOrigin(lambda);
            Vec U_hat_p_old = FindClosestPointToOrigin(lambda_p);

            Vec U_hat = BackTransform(R, L, U_hat_old);
            Vec U_hat_p = BackTransform(R_p, L_p, U_hat_p_old);

            return {U_hat, U_hat_p};
        }

        // after finding t, we get the equations of the 2 epipolar lines lambda(t) and lambda(t')
        Vec lambda=ComputeLeftEpipolarLine(best_root, f); // homogeneous coords
        Vec lambda_p=ComputeRightEpipolarLine(best_root, a, b, c, d, f_p );

        // now, we find u^ and u^' which are the points on lambda and lambda_p closest to the origin
        Vec U_hat_old=FindClosestPointToOrigin(lambda); // 2d euclidean coords
        Vec U_hat_p_old=FindClosestPointToOrigin(lambda_p); 

        // then we reverse the rigid transformations we applied at the beginning
        Vec U_hat=BackTransform(R, L, U_hat_old);
        Vec U_hat_p=BackTransform(R_p, L_p, U_hat_p_old);

        return std::pair<Vec, Vec>(U_hat, U_hat_p);
    }


    std::pair<Vec, Vec> ComputeCorrectedPairsAbs(const Vec& U, const Vec& U_prime, const Mat& F){
        
        // first we apply the rigid transformations to get F_transformed
        Mat L=TranslationMatrixToOrigin(U);
        Mat L_p=TranslationMatrixToOrigin(U_prime);

        Vec LeftEpipole=ComputeLeftEpipole(F); // homogeneous coords
        Vec RightEpipole=ComputeRightEpipole(F);

        Mat R=RotationMatrixToX(LeftEpipole);
        Mat R_p=RotationMatrixToX(RightEpipole);

        Mat F_transformed=TransformFundamentalMatrix(F, R, L, R_p, L_p);

        //F_transformed = 
        // ff'd -f'c -f'd
        // -fb    a    b
        // -fd    c    d 

        // then, we get the coefficients of F_transformed a,b,c,d,f,f_p
        double a=F_transformed(1,1);
        double b=F_transformed(1,2);
        double c=F_transformed(2,1);
        double d=F_transformed(2,2);
        double f=-(F_transformed(1,0)/b);
        double f_p=-(F_transformed(0,1)/c);

        // now, we find the best root that would minimize our cost function
        std::vector<double> roots=SolvePolyAbs(a, b, c, d, f, f_p);
        double best_root=FindBestRootAbs(roots, a, b, c, d, f, f_p);

        if (std::isinf(best_root)) {
            Vec lambda(f, 0, -1);
            Vec lambda_p(-f_p * c, a, c);

            Vec U_hat_old = FindClosestPointToOrigin(lambda);
            Vec U_hat_p_old = FindClosestPointToOrigin(lambda_p);

            Vec U_hat = BackTransform(R, L, U_hat_old);
            Vec U_hat_p = BackTransform(R_p, L_p, U_hat_p_old);

            return {U_hat, U_hat_p};
        }

        // after finding t, we get the equations of the 2 epipolar lines lambda(t) and lambda(t')
        Vec lambda=ComputeLeftEpipolarLine(best_root, f); // homogeneous coords
        Vec lambda_p=ComputeRightEpipolarLine(best_root, a, b, c, d, f_p );

        // now, we find u^ and u^' which are the points on lambda and lambda_p closest to the origin
        Vec U_hat_old=FindClosestPointToOrigin(lambda); // 2d euclidean coords
        Vec U_hat_p_old=FindClosestPointToOrigin(lambda_p); 

        // then we reverse the rigid transformations we applied at the beginning
        Vec U_hat=BackTransform(R, L, U_hat_old);
        Vec U_hat_p=BackTransform(R_p, L_p, U_hat_p_old);

        return std::pair<Vec, Vec>(U_hat, U_hat_p);
    }

} // namespace Poly