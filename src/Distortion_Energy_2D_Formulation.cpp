//
// Created by Charles Du on 4/24/23.
//

#include "Distortion_Energy_2D_Formulation.h"
#include "deformation_gradient_util.h"
#include "geo_util.h"
#include <Eigen/Eigenvalues>

Distortion_Energy_2D_Formulation::Distortion_Energy_2D_Formulation(Eigen::MatrixXd restV_,
                                                                   Eigen::Matrix2Xd initV, Eigen::Matrix3Xi F_,
                                                                   const Eigen::VectorXi &handles)
                                                                   : Injective_Energy_2D_Formulation(std::move(restV_), std::move(initV), std::move(F_), handles) {
    // compute rest area
    restA.resize(F.cols());
    for (int i = 0; i < F.cols(); ++i) {
        auto v1 = restV.col(F(0, i));
        auto v2 = restV.col(F(1, i));
        auto v3 = restV.col(F(2, i));
        auto e1 = v2 - v3;
        auto e2 = v3 - v1;
        auto e3 = v1 - v2;
        restA(i) = compute_Heron_tri_area(e1.squaredNorm(), e2.squaredNorm(), e3.squaredNorm());
    }

    // compute inverse of edge matrix and pFpx
    rest_invEdgeMat.reserve(F.cols());
    Eigen::Matrix2d rest_edge_mat;
    for (int i = 0; i < F.cols(); ++i) {
        rest_invEdgeMat.emplace_back();
        compute_edge_matrix(restV.col(F(0, i)),
                            restV.col(F(1, i)),
                            restV.col(F(2, i)),
                            rest_edge_mat);
        rest_invEdgeMat.back() = rest_edge_mat.inverse();
    }

    // compute derivative of deformation gradient w.r.t. target vertices
    pFpx_list.reserve(F.cols());
    for (int i = 0; i < F.cols(); ++i) {
        pFpx_list.emplace_back();
        compute_flatten_pFpx(rest_invEdgeMat[i], pFpx_list.back());
    }

}

double Distortion_Energy_2D_Formulation::compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) {
    update_x_and_V(x);

    energy_list.resize(free_faceI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);

        energy_list(i) = compute_triangle_energy(V.col(F(0, fi)),
                                                 V.col(F(1, fi)),
                                                 V.col(F(2, fi)),
                                                 restA(fi),
                                                 rest_invEdgeMat[fi]);
    }

    return energy_list.sum();
}

double Distortion_Energy_2D_Formulation::compute_triangle_energy(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2,
                                                                 const Eigen::Vector2d &v3, double rest_triangle_area,
                                                                 const Eigen::Matrix2d &rest_inv_EdgeMat) {
    // compute deformation gradient
    Eigen::Matrix2d target_edge_matrix;
    compute_edge_matrix(v1, v2, v3, target_edge_matrix);
    Eigen::Matrix2d deformation_gradient = target_edge_matrix * rest_inv_EdgeMat;
    // compute SVD
    Eigen::Matrix2d U, V;
    Eigen::Vector2d singular_values;
    compute_SVD(deformation_gradient, U, singular_values, V);
    double s1 = singular_values[0];
    double s2 = singular_values[1];
    // S-centric invariants
    double I1 = s1 + s2;
    double I2 = s1 * s1 + s2 * s2;
    double I3 = s1 * s2;
    // compute energy
    return rest_triangle_area * compute_psi(I1, I2, I3);
}

double Distortion_Energy_2D_Formulation::compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    update_x_and_V(x);

    Eigen::VectorXd energyList(free_faceI.size());
    std::vector<Eigen::Matrix2Xd> gradList(free_faceI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);
        energyList(i) = compute_triangle_energy_with_gradient(V.col(F(0, fi)),
                                                             V.col(F(1, fi)),
                                                             V.col(F(2, fi)),
                                                             restA(fi),
                                                             rest_invEdgeMat[fi],
                                                             pFpx_list[fi],
                                                             gradList[i]);
    }

    // accumulate gradient
    Eigen::Matrix2Xd grad_vert = Eigen::Matrix2Xd::Zero(2, V.cols()); // 2D vertex
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);

        grad_vert.col(F(0, fi)) += gradList[i].col(0);
        grad_vert.col(F(1, fi)) += gradList[i].col(1);
        grad_vert.col(F(2, fi)) += gradList[i].col(2);
    }

    // flattened gradient of free vertices
    grad.resize(freeI.size() * grad_vert.rows());
    for (int i = 0; i < freeI.size(); ++i) {
        grad(2 * i) = grad_vert(0, freeI(i));
        grad(2 * i + 1) = grad_vert(1, freeI(i));
    }

    return energyList.sum();

}

double Distortion_Energy_2D_Formulation::compute_triangle_energy_with_gradient(const Eigen::Vector2d &v1,
                                                                               const Eigen::Vector2d &v2,
                                                                               const Eigen::Vector2d &v3, double rest_triangle_area,
                                                                               const Eigen::Matrix2d &rest_inv_EdgeMat,
                                                                               const Eigen::MatrixXd &pFpx,
                                                                               Eigen::Matrix2Xd &grad) {
    // compute deformation gradient
    Eigen::Matrix2d target_edge_matrix;
    compute_edge_matrix(v1, v2, v3, target_edge_matrix);
    Eigen::Matrix2d deformation_gradient = target_edge_matrix * rest_inv_EdgeMat;
    // compute SVD
    Eigen::Matrix2d U, V;
    Eigen::Vector2d singular_values;
    compute_SVD(deformation_gradient, U, singular_values, V);
    double s1 = singular_values[0];
    double s2 = singular_values[1];
    // S-centric invariants
    double I1 = s1 + s2;
    double I2 = s1 * s1 + s2 * s2;
    double I3 = s1 * s2;
    // compute psi and its gradient w.r.t. I1, I2, I3
    Eigen::Vector3d grad_psi;
    double psi = compute_psi_with_gradient(I1, I2, I3, grad_psi);
    // compute gradient of psi w.r.t. flattened deformation gradient
    Eigen::Vector4d f;
    vectorize2x2(deformation_gradient, f);
    Eigen::Vector4d d_I1_d_f;
    compute_d_I1_d_f(U, V, d_I1_d_f);
    Eigen::Vector4d d_I2_d_f = compute_d_I2_d_f(f);
    Eigen::Vector4d d_I3_d_f;
    compute_d_I3_d_f(U, singular_values, V, d_I3_d_f);
    Eigen::Vector4d d_psi_d_f = grad_psi.x() * d_I1_d_f + grad_psi.y() * d_I2_d_f + grad_psi.z() * d_I3_d_f;
    // compute gradient of energy
    Eigen::Vector<double, 6> d_psi_d_x = pFpx.transpose() * d_psi_d_f;
    grad.resize(2, 3);
    grad.col(0) = d_psi_d_x.segment<2>(0);
    grad.col(1) = d_psi_d_x.segment<2>(2);
    grad.col(2) = d_psi_d_x.segment<2>(4);
    grad *= rest_triangle_area;
    // return energy
    return rest_triangle_area * psi;
}

double Distortion_Energy_2D_Formulation::compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x,
                                                                              Eigen::VectorXd &energy_list,
                                                                              Eigen::VectorXd &grad, SpMat &Hess) {
    update_x_and_V(x);

    energy_list.resize(free_faceI.size());
    std::vector<Eigen::Matrix2Xd> gradList(free_faceI.size());

    const int vDim = 2;
    const int simplex_size = 3;
    std::vector<Eigen::Triplet<double>> tripletList(simplex_size * simplex_size * vDim * vDim * free_faceI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);
        Eigen::MatrixXd hess;
        energy_list(i) = compute_triangle_energy_with_gradient_projected_Hessian(
                V.col(F(0, fi)), V.col(F(1, fi)), V.col(F(2, fi)),
                restA(fi),
                rest_invEdgeMat[fi],
                pFpx_list[fi],
                gradList[i], hess);

        // update Hessian of free vertices
        int current_index = i * simplex_size * simplex_size * vDim * vDim;
        Eigen::Vector3i indices = F_free.col(fi);
        for (int j = 0; j < simplex_size; ++j) {
            int idx_j = indices(j);
            for (int k = 0; k < simplex_size; ++k) {
                int idx_k = indices(k);
                if (idx_j != -1 && idx_k != -1) {
                    for (int l = 0; l < vDim; ++l) {
                        for (int n = 0; n < vDim; ++n) {
                            tripletList[current_index] = Eigen::Triplet<double>(idx_j * vDim + l, idx_k * vDim + n,
                                                                                hess(j * vDim + l, k * vDim + n));
                            ++current_index;
                        }
                    }
                }
            }
        }

    }

    // add small positive values to the diagonal of Hessian
    for (auto i = 0; i < vDim * freeI.size(); ++i) {
        tripletList.emplace_back(i, i, 1e-8);
    }

    // get Hessian on free vertices
    Hess.resize(vDim * freeI.size(), vDim * freeI.size());
    Hess.setFromTriplets(tripletList.begin(), tripletList.end());


    // accumulate gradient
    Eigen::Matrix2Xd grad_vert = Eigen::Matrix2Xd::Zero(vDim, V.cols());
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);

        grad_vert.col(F(0, fi)) += gradList[i].col(0);
        grad_vert.col(F(1, fi)) += gradList[i].col(1);
        grad_vert.col(F(2, fi)) += gradList[i].col(2);
    }

    // flattened gradient of free vertices
    grad.resize(freeI.size() * grad_vert.rows());
    for (int i = 0; i < freeI.size(); ++i) {
        grad(2 * i) = grad_vert(0, freeI(i));
        grad(2 * i + 1) = grad_vert(1, freeI(i));
    }

    return energy_list.sum();
}

double
Distortion_Energy_2D_Formulation::compute_triangle_energy_with_gradient_projected_Hessian(const Eigen::Vector2d &v1,
                                                                                          const Eigen::Vector2d &v2,
                                                                                          const Eigen::Vector2d &v3,
                                                                                          double rest_triangle_area,
                                                                                          const Eigen::Matrix2d &rest_inv_EdgeMat,
                                                                                          const Eigen::MatrixXd &pFpx,
                                                                                          Eigen::Matrix2Xd &grad,
                                                                                          Eigen::MatrixXd &hess) {
    // compute deformation gradient
    Eigen::Matrix2d target_edge_matrix;
    compute_edge_matrix(v1, v2, v3, target_edge_matrix);
    Eigen::Matrix2d deformation_gradient = target_edge_matrix * rest_inv_EdgeMat;
    // compute SVD
    Eigen::Matrix2d U, V;
    Eigen::Vector2d singular_values;
    compute_SVD(deformation_gradient, U, singular_values, V);
    double s1 = singular_values[0];
    double s2 = singular_values[1];
    // S-centric invariants
    double I1 = s1 + s2;
    double I2 = s1 * s1 + s2 * s2;
    double I3 = s1 * s2;
    // compute psi and its gradient w.r.t. I1, I2, I3
    Eigen::Vector3d grad_psi;
    double psi = compute_psi_with_gradient(I1, I2, I3, grad_psi);
    // compute gradient of psi w.r.t. flattened deformation gradient
    Eigen::Vector4d f;
    vectorize2x2(deformation_gradient, f);
    Eigen::Vector4d d_I1_d_f;
    compute_d_I1_d_f(U, V, d_I1_d_f);
    Eigen::Vector4d d_I2_d_f = compute_d_I2_d_f(f);
    Eigen::Vector4d d_I3_d_f;
    compute_d_I3_d_f(U, singular_values, V, d_I3_d_f);
    Eigen::Vector4d d_psi_d_f = grad_psi.x() * d_I1_d_f + grad_psi.y() * d_I2_d_f + grad_psi.z() * d_I3_d_f;
    // compute gradient of energy
    Eigen::Vector<double, 6> d_psi_d_x = pFpx.transpose() * d_psi_d_f;
    grad.resize(2, 3);
    grad.col(0) = d_psi_d_x.segment<2>(0);
    grad.col(1) = d_psi_d_x.segment<2>(2);
    grad.col(2) = d_psi_d_x.segment<2>(4);
    grad *= rest_triangle_area;
    // compute Hessian
    Eigen::Vector4d lambdas;
    Eigen::Matrix2d matA;
    bool scale_decoupled = compute_analytic_eigen_information(s1, s2, I1, I2, I3, lambdas, matA);
    //// twist mode
    double eigen_value_twist = lambdas[2];
    Eigen::Vector4d eigen_vec_twist;
    vectorize2x2((U.col(1) * V.col(0).transpose() - U.col(0) * V.col(1).transpose()) / sqrt(2.),
                 eigen_vec_twist);
    //// flip mode
    double eigen_value_flip = lambdas[3];
    Eigen::Vector4d eigen_vec_flip;
    vectorize2x2((U.col(1) * V.col(0).transpose() + U.col(0) * V.col(1).transpose()) / sqrt(2.),
                 eigen_vec_flip);
    //// scale mode
    Eigen::Vector4d vec_d1, vec_d2;
    vectorize2x2(U.col(0) * V.col(0).transpose(), vec_d1);
    vectorize2x2(U.col(1) * V.col(1).transpose(), vec_d2);
    double eigen_value_scale1;
    double eigen_value_scale2;
    Eigen::Vector4d eigen_vec_scale1;
    Eigen::Vector4d eigen_vec_scale2;
    if (scale_decoupled) {
        eigen_value_scale1 = lambdas[0];
        eigen_value_scale2 = lambdas[1];
        eigen_vec_scale1 = vec_d1;
        eigen_vec_scale2 = vec_d2;
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigenSolver(matA);
        Eigen::Vector2d matA_eigen_values = eigenSolver.eigenvalues();
        eigen_value_scale1 = matA_eigen_values[0];
        eigen_value_scale2 = matA_eigen_values[1];
        double a12 = matA(0, 1);
        double a22 = matA(1, 1);
        double beta = (eigen_value_scale1 - a22) / a12;
        double norm_beta_1 = sqrt(1. + beta * beta);
        eigen_vec_scale1 = (beta * vec_d1 + vec_d2) / norm_beta_1;
        eigen_vec_scale2 = (vec_d1 - beta * vec_d2) / norm_beta_1;
    }
    //// PSD f-Hessian
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
    //// PSD x-Hessian
    hess = pFpx.transpose() * (rest_triangle_area * f_Hess) * pFpx;
    // return energy
    return rest_triangle_area * psi;
}
