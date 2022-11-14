//
// Created by Charles Du on 11/13/22.
//

#include "IsoTLC_2D_Formulation.h"
#include "geo_util.h"
#include "deformation_gradient_util.h"
#include <utility>
#include <Eigen/Eigenvalues>


IsoTLC_2D_Formulation::IsoTLC_2D_Formulation(const Eigen::MatrixXd &restV, Eigen::Matrix2Xd initV, Eigen::Matrix3Xi F_,
                                             const Eigen::VectorXi &handles, std::string form_, double alpha_)
        : TLC_2D_Formulation(restV, std::move(initV), std::move(F_), handles, std::move(form_), alpha_) {
    // compute scaled squared areas of auxiliary triangles: (alpha/2 + alpha^2)*(A_rest)^2
    // lambda1 = lambda2 = 1/4, k = 1
    scaled_squared_restA.resize(F.cols());
    for (int i = 0; i < F.cols(); ++i) {
        scaled_squared_restA(i) = restA(i) * restA(i);
    }
    scaled_squared_restA *= alpha * (0.5 + alpha);

    // compute scaled Dirichlet coefficients
    scaled_Dirichlet_coefficients.resize(3, F.cols());
    for (int i = 0; i < F.cols(); ++i) {
        double d1 = restD(0, i);
        double d2 = restD(1, i);
        double d3 = restD(2, i);
        scaled_Dirichlet_coefficients(0, i) = d2 + d3 - d1;
        scaled_Dirichlet_coefficients(1, i) = d3 + d1 - d2;
        scaled_Dirichlet_coefficients(2, i) = d1 + d2 - d3;
    }
    scaled_Dirichlet_coefficients *= (alpha / 16);

    squared_targetA_scale_coeff = 1 + 0.5 * alpha;

    // coefficients for analytic eigen system of Hessian
    half_alpha = 0.5 * alpha;
    one_plus_half_alpha = 1 + 0.5 * alpha;
    coeff_diag = alpha * (0.5 + alpha);
    coeff_off_diag = alpha * (1 + 2.25 * alpha + alpha * alpha);
}

double IsoTLC_2D_Formulation::compute_lifted_TriArea(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2,
                                                     const Eigen::Vector2d &v3,
                                                     const Eigen::Vector3d &Dirichlet_coefficient,
                                                     double scaled_squared_rest_area, double &signed_area) const {
    signed_area = compute_tri_signed_area(v1, v2, v3);

    return sqrt(
            squared_targetA_scale_coeff * signed_area * signed_area + Dirichlet_coefficient(0) * (v2 - v3).squaredNorm()
            + Dirichlet_coefficient(1) * (v3 - v1).squaredNorm() + Dirichlet_coefficient(2) * (v1 - v2).squaredNorm()
            + scaled_squared_rest_area
    );
}

double
IsoTLC_2D_Formulation::compute_residual(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2, const Eigen::Vector2d &v3,
                                        const Eigen::Vector3d &Dirichlet_coefficient, double scaled_squared_rest_area,
                                        double &signed_area) const {
    signed_area = compute_tri_signed_area(v1, v2, v3);

    return sqrt(
            squared_targetA_scale_coeff * signed_area * signed_area + Dirichlet_coefficient(0) * (v2 - v3).squaredNorm()
            + Dirichlet_coefficient(1) * (v3 - v1).squaredNorm() + Dirichlet_coefficient(2) * (v1 - v2).squaredNorm()
            + scaled_squared_rest_area
    ) - signed_area;
}

double IsoTLC_2D_Formulation::compute_lifted_TriArea_with_gradient(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2,
                                                                   const Eigen::Vector2d &v3,
                                                                   const Eigen::Vector3d &Dirichlet_coefficient,
                                                                   double scaled_squared_rest_area,
                                                                   Eigen::Matrix2Xd &grad, double &signed_area) const {
    Eigen::Vector2d e1 = v3 - v2;
    Eigen::Vector2d e2 = v1 - v3;
    Eigen::Vector2d e3 = v2 - v1;

    double d1 = Dirichlet_coefficient(0);
    double d2 = Dirichlet_coefficient(1);
    double d3 = Dirichlet_coefficient(2);

    signed_area = compute_tri_signed_area(v1, v2, v3);
    double energy = sqrt(squared_targetA_scale_coeff * signed_area * signed_area + d1 * e1.squaredNorm()
                         + d2 * e2.squaredNorm() + d3 * e3.squaredNorm()
                         + scaled_squared_rest_area);

    double f1 = one_plus_half_alpha * signed_area / 2;

//    grad.resize(vert.rows(), vert.cols());
    grad.resize(2, 3);
    grad.col(0) = d2 * e2 - d3 * e3 + f1 * rotate_90deg(e1);
    grad.col(1) = d3 * e3 - d1 * e1 + f1 * rotate_90deg(e2);
    grad.col(2) = d1 * e1 - d2 * e2 + f1 * rotate_90deg(e3);

    grad /= energy;

    return energy;
}

double IsoTLC_2D_Formulation::compute_lifted_TriArea_with_gradient_projected_residual_Hessian(const Eigen::Vector2d &v1,
                                                                                              const Eigen::Vector2d &v2,
                                                                                              const Eigen::Vector2d &v3,
                                                                                              const Eigen::Vector3d &Dirichlet_coefficient,
                                                                                              double scaled_squared_rest_area,
                                                                                              double rest_area,
                                                                                              const Eigen::Matrix2d &rest_inverse_EdgeMat,
                                                                                              const Eigen::MatrixXd &pFpx,
                                                                                              Eigen::Matrix2Xd &grad,
                                                                                              Eigen::MatrixXd &Hess,
                                                                                              double &signed_area) const {
    Eigen::Vector2d e1 = v3 - v2;
    Eigen::Vector2d e2 = v1 - v3;
    Eigen::Vector2d e3 = v2 - v1;

    double d1 = Dirichlet_coefficient(0);
    double d2 = Dirichlet_coefficient(1);
    double d3 = Dirichlet_coefficient(2);

    signed_area = compute_tri_signed_area(v1, v2, v3);
    double energy = sqrt(squared_targetA_scale_coeff * signed_area * signed_area + d1 * e1.squaredNorm()
                         + d2 * e2.squaredNorm() + d3 * e3.squaredNorm()
                         + scaled_squared_rest_area);

    double f1 = one_plus_half_alpha * signed_area / 2;

//    grad.resize(vert.rows(), vert.cols());
    grad.resize(2, 3);
    grad.col(0) = d2 * e2 - d3 * e3 + f1 * rotate_90deg(e1);
    grad.col(1) = d3 * e3 - d1 * e1 + f1 * rotate_90deg(e2);
    grad.col(2) = d1 * e1 - d2 * e2 + f1 * rotate_90deg(e3);

    grad /= energy;

    // PSD projected Hessian of (isometric lifted triangle area - signed triangle area)
    Eigen::Matrix2d target_edge_matrix;
    compute_edge_matrix(v1, v2, v3, target_edge_matrix);
//    Eigen::Matrix2d deformation_gradient = target_edge_matrix * rest_inverse_EdgeMat;
    Eigen::Matrix2d U, V;
    Eigen::Vector2d singular_values;
//    compute_SVD(deformation_gradient, U, singular_values, V);
    compute_SVD(target_edge_matrix * rest_inverse_EdgeMat, U, singular_values, V);
    double s1 = singular_values[0];
    double s2 = singular_values[1];
    // S-centric invariants
//    double I1 = s1 + s2;
    double I2 = s1 * s1 + s2 * s2;
    double I3 = s1 * s2;
    // Analytic Eigen-system of f-Hessian
    double energy_density = energy / rest_area;
    //// twist
    double eigen_value_twist = (half_alpha + I3 * one_plus_half_alpha) / energy_density - 1;
    Eigen::Vector4d eigen_vec_twist;
    vectorize2x2((U.col(1) * V.col(0).transpose() - U.col(0) * V.col(1).transpose()) / sqrt(2.),
                 eigen_vec_twist);
    //// flip
    double eigen_value_flip = (half_alpha - I3 * one_plus_half_alpha) / energy_density + 1;
    Eigen::Vector4d eigen_vec_flip;
    vectorize2x2((U.col(1) * V.col(0).transpose() + U.col(0) * V.col(1).transpose()) / sqrt(2.),
                 eigen_vec_flip);
    //// scale
    Eigen::Vector4d vec_d1, vec_d2;
    vectorize2x2(U.col(0) * V.col(0).transpose(), vec_d1);
    vectorize2x2(U.col(1) * V.col(1).transpose(), vec_d2);

    double energy_cube = energy_density * energy_density * energy_density;
    double a11 = (coeff_diag + half_alpha * s2 * s2) * (half_alpha + one_plus_half_alpha * s2 * s2) / energy_cube;
    double a22 = (coeff_diag + half_alpha * s1 * s1) * (half_alpha + one_plus_half_alpha * s1 * s1) / energy_cube;
    double a12 = (coeff_off_diag * I3 + one_plus_half_alpha * I3 * (one_plus_half_alpha * I3 * I3 + half_alpha * I2)) /
                 energy_cube - 1;
    double eigen_value_scale1;
    double eigen_value_scale2;
    Eigen::Vector4d eigen_vec_scale1;
    Eigen::Vector4d eigen_vec_scale2;
    if (a12 == 0) {
        eigen_value_scale1 = a11;
        eigen_value_scale2 = a22;
        eigen_vec_scale1 = vec_d1;
        eigen_vec_scale2 = vec_d2;
    } else {
        Eigen::Matrix2d matA;
        matA << a11, a12,
                a12, a22;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(matA);
        Eigen::Vector2d matA_eigen_values = eigenSolver.eigenvalues();
        eigen_value_scale1 = matA_eigen_values[0];
        eigen_value_scale2 = matA_eigen_values[1];
        /*eigen_value_scale1 = (a11 + a12 - sqrt((a11-a22)*(a11-a22)+4*a12*a12))/2;
        eigen_value_scale2 = (a11 + a12 + sqrt((a11-a22)*(a11-a22)+4*a12*a12))/2;*/
        double beta = (eigen_value_scale1 - a22) / a12;
        double norm_beta_1 = sqrt(1. + beta * beta);
        eigen_vec_scale1 = (beta * vec_d1 + vec_d2) / norm_beta_1;
        eigen_vec_scale2 = (vec_d1 - beta * vec_d2) / norm_beta_1;
    }
    // PSD f-Hessian
    Eigen::Matrix4d f_Hess;
    f_Hess.setZero();
    if (eigen_value_twist > 0) {
        f_Hess += eigen_value_twist * eigen_vec_twist * eigen_vec_twist.transpose();
    }
    if (eigen_value_flip > 0) {
        f_Hess += eigen_value_flip * eigen_vec_flip * eigen_vec_flip.transpose();
    }
    if (eigen_value_scale1 > 0) {
        f_Hess += eigen_value_scale1 * eigen_vec_scale1 * eigen_vec_scale1.transpose();
    }
    if (eigen_value_scale2 > 0) {
        f_Hess += eigen_value_scale2 * eigen_vec_scale2 * eigen_vec_scale2.transpose();
    }
    // PSD x-Hessian
    Hess = pFpx.transpose() * (rest_area * f_Hess) * pFpx;

    return energy;
}


