//
// Created by Charles Du on 11/12/22.
//

#include "TLC_2D_Formulation.h"
#include "geo_util.h"
#include "deformation_gradient_util.h"
#include <iostream>
#include <utility>
#include <Eigen/Eigenvalues>


using namespace Eigen;

TLC_2D_Formulation::TLC_2D_Formulation(const Eigen::MatrixXd &restV, Eigen::Matrix2Xd initV,
                                       Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles, std::string form_,
                                       double alpha_) : Injective_Energy_2D_Formulation(restV, std::move(initV),
                                                                                        std::move(F_), handles),
                                                        form(std::move(form_)), alpha(alpha_) {
    // print formulation parameters
    std::cout << "form: " << form << std::endl;
    std::cout << "alpha: " << alpha << std::endl;

    // initialize TLC
    initialize();

    // precompute some terms in the TLC energy
    // compute scaled squared areas of auxiliary triangles: alpha^2*A_rest^2
    scaled_squared_restA.resize(F.cols());
    for (int i = 0; i < F.cols(); ++i) {
        scaled_squared_restA(i) = restA(i) * restA(i);
    }
    scaled_squared_restA *= (alpha * alpha);

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
    scaled_Dirichlet_coefficients *= (alpha / 8);
}

void TLC_2D_Formulation::initialize() {
    // compute restD
    compute_squared_edge_Length(restV, F, restD);

    // compute rest area
    restA.resize(F.cols());
    for (int i = 0; i < F.cols(); ++i) {
        restA(i) = compute_Heron_tri_area(restD(0, i), restD(1, i), restD(2, i));
    }
    double total_rest_area = restA.sum();

    if (form != "harmonic") { // Tutte form
        // we assume the auxiliary triangles are equilateral triangles
        // whose total measure is the same as the input rest mesh.
        double regular_tri_area = total_rest_area / F.cols();
        double regular_tri_squared_length = 4 * regular_tri_area / sqrt(3.0);
        restA = Eigen::VectorXd::Constant(F.cols(), regular_tri_area);
        restD = Eigen::MatrixXd::Constant(3, F.cols(), regular_tri_squared_length);
        // compute inverse of edge matrix and pFpx
        Eigen::Vector2d v1(0, 0);
        double regular_tri_edge_length = sqrt(regular_tri_squared_length);
        Eigen::Vector2d v2(regular_tri_edge_length, 0);
        Eigen::Vector2d v3(regular_tri_edge_length / 2, sqrt(3.) * regular_tri_edge_length / 2);
        Eigen::Matrix2d edgeMat;
        compute_edge_matrix(v1, v2, v3, edgeMat);
        Eigen::Matrix2d invEdgeMat = edgeMat.inverse();
        Eigen::MatrixXd pFpx;
        compute_flatten_pFpx(invEdgeMat, pFpx);
        rest_invEdgeMat.resize(F.cols(), invEdgeMat);
        pFpx_list.resize(F.cols(), pFpx);
    } else { // harmonic form: use rest triangles as auxiliary triangles
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
}

double TLC_2D_Formulation::compute_energy(const VectorXd &x, VectorXd &energy_list) {
    update_x_and_V(x);
    if (fixed_boundary) { // TLC
        return compute_total_lifted_content(V, energy_list);
    } else { // TLC - total signed area
        return compute_total_residual(V, energy_list);
    }
}

double TLC_2D_Formulation::compute_energy_with_gradient(const VectorXd &x, VectorXd &grad) {
    update_x_and_V(x);

    // TLC
    Matrix2Xd tlc_grad;
    double tlc_energy = compute_total_lifted_content_with_gradient(V, tlc_grad);

    if (fixed_boundary) {
        // gradient of free vertices
        grad.resize(freeI.size() * tlc_grad.rows());
        for (int i = 0; i < freeI.size(); ++i) {
            grad(2 * i) = tlc_grad(0, freeI(i));
            grad(2 * i + 1) = tlc_grad(1, freeI(i));
        }
        // energy
        return tlc_energy;
    } else { // boundary is not fixed, compute gradient of total residual
        std::vector<Point> boundary_vertices(boundaryI.size());
        for (int i = 0; i < boundaryI.size(); ++i) {
            boundary_vertices[i] = V.col(boundaryI(i));
        }
        Matrix2Xd total_signed_area_grad;
        total_signed_area_grad.setZero(2, boundaryI.size());
        compute_total_signed_area_with_gradient(
                boundary_vertices, boundary_edges, total_signed_area_grad);

        // gradient of (TLC - total signed area)
        Matrix2Xd &m_grad = tlc_grad;
        for (int i = 0; i < boundaryI.size(); ++i) {
            m_grad.col(boundaryI(i)) -= total_signed_area_grad.col(i);
        }

        // gradient of free vertices
        grad.resize(freeI.size() * m_grad.rows());
        for (int i = 0; i < freeI.size(); ++i) {
            grad(2 * i) = m_grad(0, freeI(i));
            grad(2 * i + 1) = m_grad(1, freeI(i));
        }

        // energy
        return tlc_energy - free_face_areas.sum();
    }
}

double
TLC_2D_Formulation::compute_energy_with_gradient_Hessian(const VectorXd &x, VectorXd &energy_list, VectorXd &grad,
                                                         SpMat &Hess) {
    update_x_and_V(x);

    // TLC energy of free faces, TLC full gradient,
    // and PSD projected Hessian of (TLC - total signed area) on free vertices
    VectorXd &lifted_content_list = energy_list;
    Matrix2Xd tlc_grad;
    double tlc_energy = compute_total_lifted_content_with_gradient_and_projectedHessian(V, freeI,
                                                                                        F_free,
                                                                                        lifted_content_list,
                                                                                        tlc_grad,
                                                                                        Hess);
    if (fixed_boundary) {
        // gradient of free vertices
        grad.resize(freeI.size() * tlc_grad.rows());
        for (int i = 0; i < freeI.size(); ++i) {
            grad(2 * i) = tlc_grad(0, freeI(i));
            grad(2 * i + 1) = tlc_grad(1, freeI(i));
        }
        // energy
        return tlc_energy;
    } else { // boundary is not fixed
        // energy_list: list of residuals
        energy_list -= free_face_areas;
        // compute gradient of total signed area
        std::vector<Point> boundary_vertices(boundaryI.size());
        for (int i = 0; i < boundaryI.size(); ++i) {
            boundary_vertices[i] = V.col(boundaryI(i));
        }
        Matrix2Xd total_signed_area_grad;
        total_signed_area_grad.setZero(2, boundaryI.size());
        compute_total_signed_area_with_gradient(
                boundary_vertices, boundary_edges, total_signed_area_grad);
        // gradient of (TLC - total signed area)
        Matrix2Xd &m_grad = tlc_grad;
        for (int i = 0; i < boundaryI.size(); ++i) {
            m_grad.col(boundaryI(i)) -= total_signed_area_grad.col(i);
        }
        // gradient of free vertices
        grad.resize(freeI.size() * m_grad.rows());
        for (int i = 0; i < freeI.size(); ++i) {
            grad(2 * i) = m_grad(0, freeI(i));
            grad(2 * i + 1) = m_grad(1, freeI(i));
        }
        // energy
        return energy_list.sum();
    }
}


double TLC_2D_Formulation::compute_total_lifted_content(const Eigen::Matrix2Xd &vertices,
                                                        Eigen::VectorXd &energyList) {
    energyList.resize(free_faceI.size());
//    int vDim = 2;
//    int simplex_size = 3; //triangle

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);

        energyList(i) = compute_lifted_TriArea(vertices.col(F(0, fi)),
                                               vertices.col(F(1, fi)),
                                               vertices.col(F(2, fi)),
                                               scaled_Dirichlet_coefficients.col(fi),
                                               scaled_squared_restA(fi),
                                               free_face_areas[i]);
    }

    return energyList.sum();
}

double TLC_2D_Formulation::compute_total_residual(const Matrix2Xd &vertices, VectorXd &residual_list) {
    residual_list.resize(free_faceI.size());
//    int vDim = 2;
//    int simplex_size = 3; //triangle

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);

        residual_list(i) = compute_residual(vertices.col(F(0, fi)),
                                            vertices.col(F(1, fi)),
                                            vertices.col(F(2, fi)),
                                            scaled_Dirichlet_coefficients.col(fi),
                                            scaled_squared_restA(fi),
                                            free_face_areas[i]);
    }

    return residual_list.sum();
}

double
TLC_2D_Formulation::compute_total_lifted_content_with_gradient(const Matrix2Xd &vertices, Matrix2Xd &grad) {
    //    int vDim = 2;
    Eigen::VectorXd energyList(free_faceI.size());
//    energyList.resize(free_faceI.size());
    std::vector<Eigen::Matrix2Xd> gradList(free_faceI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);
        energyList(i) = compute_lifted_TriArea_with_gradient(vertices.col(F(0, fi)),
                                                             vertices.col(F(1, fi)),
                                                             vertices.col(F(2, fi)),
                                                             scaled_Dirichlet_coefficients.col(fi),
                                                             scaled_squared_restA(fi),
                                                             gradList[i],
                                                             free_face_areas[i]);
    }

    // accumulate gradient
    grad = Eigen::Matrix2Xd::Zero(2, vertices.cols());
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);

        grad.col(F(0, fi)) += gradList[i].col(0);
        grad.col(F(1, fi)) += gradList[i].col(1);
        grad.col(F(2, fi)) += gradList[i].col(2);
    }

    return energyList.sum();
}

double TLC_2D_Formulation::compute_total_lifted_content_with_gradient_and_projectedHessian(const Matrix2Xd &vertices,
                                                                                           const VectorXi &freeI,
                                                                                           const Matrix3Xi &F_free,
                                                                                           VectorXd &lifted_content_list,
                                                                                           Matrix2Xd &grad,
                                                                                           SpMat &Hess) {
    int vDim = 2;
//    lifted_content_list.resize(F.cols());
    lifted_content_list.resize(free_faceI.size());
//    std::vector<Eigen::Matrix2Xd> gradList(F.cols());
    std::vector<Eigen::Matrix2Xd> gradList(free_faceI.size());
    grad = Eigen::Matrix2Xd::Zero(2, vertices.cols());

//    std::vector<Eigen::Triplet<double>> tripletList(3 * 3 * vDim * vDim * F.cols());
    std::vector<Eigen::Triplet<double>> tripletList(3 * 3 * vDim * vDim * free_faceI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_faceI.size(); ++i) {
        auto fi = free_faceI(i);
        Eigen::MatrixXd hess;
        lifted_content_list(i) = compute_lifted_TriArea_with_gradient_projected_residual_Hessian(
                vertices.col(F(0, fi)), vertices.col(F(1, fi)), vertices.col(F(2, fi)),
                scaled_Dirichlet_coefficients.col(fi),
                scaled_squared_restA(fi),
                restA(fi),
                rest_invEdgeMat[fi],
                pFpx_list[fi],
                gradList[i], hess,
                free_face_areas[i]);

        // update Hessian of free vertices
        int current_index = i * 3 * 3 * vDim * vDim;
        Eigen::Vector3i indices = F_free.col(fi);
        for (int j = 0; j < 3; ++j) {
            int idx_j = indices(j);
            for (int k = 0; k < 3; ++k) {
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

    // get gradient
    for (int i = 0; i < free_faceI.size(); i++) {
        auto fi = free_faceI(i);
        grad.col(F(0, fi)) += gradList[i].col(0);
        grad.col(F(1, fi)) += gradList[i].col(1);
        grad.col(F(2, fi)) += gradList[i].col(2);
    }

    // add small positive values to the diagonal of Hessian
    for (auto i = 0; i < vDim * freeI.size(); ++i) {
        tripletList.emplace_back(i, i, 1e-8);
    }

    // get Hessian on free vertices
    Hess.resize(vDim * freeI.size(), vDim * freeI.size());
    Hess.setFromTriplets(tripletList.begin(), tripletList.end());

    return lifted_content_list.sum();
}


double TLC_2D_Formulation::compute_lifted_TriArea(const Eigen::Vector2d &v1,
                                                  const Eigen::Vector2d &v2,
                                                  const Eigen::Vector2d &v3,
                                                  const Eigen::Vector3d &Dirichlet_coefficient,
                                                  double scaled_squared_rest_area,
                                                  double &signed_area) const {
    signed_area = compute_tri_signed_area(v1, v2, v3);

    return sqrt(signed_area * signed_area + Dirichlet_coefficient(0) * (v2 - v3).squaredNorm()
                + Dirichlet_coefficient(1) * (v3 - v1).squaredNorm() +
                Dirichlet_coefficient(2) * (v1 - v2).squaredNorm()
                + scaled_squared_rest_area
    );
}

double TLC_2D_Formulation::compute_residual(const Eigen::Vector2d &v1,
                                            const Eigen::Vector2d &v2,
                                            const Eigen::Vector2d &v3,
                                            const Eigen::Vector3d &Dirichlet_coefficient,
                                            double scaled_squared_rest_area,
                                            double &signed_area) const {
    signed_area = compute_tri_signed_area(v1, v2, v3);

    return sqrt(signed_area * signed_area + Dirichlet_coefficient(0) * (v2 - v3).squaredNorm()
                + Dirichlet_coefficient(1) * (v3 - v1).squaredNorm() +
                Dirichlet_coefficient(2) * (v1 - v2).squaredNorm()
                + scaled_squared_rest_area
    ) - signed_area;
}

double TLC_2D_Formulation::compute_lifted_TriArea_with_gradient(const Eigen::Vector2d &v1,
                                                                const Eigen::Vector2d &v2,
                                                                const Eigen::Vector2d &v3,
                                                                const Eigen::Vector3d &Dirichlet_coefficient,
                                                                double scaled_squared_rest_area,
                                                                Eigen::Matrix2Xd &grad,
                                                                double &signed_area) const {
    Eigen::Vector2d e1 = v3 - v2;
    Eigen::Vector2d e2 = v1 - v3;
    Eigen::Vector2d e3 = v2 - v1;

    double d1 = Dirichlet_coefficient(0);
    double d2 = Dirichlet_coefficient(1);
    double d3 = Dirichlet_coefficient(2);

    signed_area = compute_tri_signed_area(v1, v2, v3);
    double energy = sqrt(signed_area * signed_area + d1 * e1.squaredNorm()
                         + d2 * e2.squaredNorm() + d3 * e3.squaredNorm()
                         + scaled_squared_rest_area);

    double f1 = signed_area / 2;

//    grad.resize(vert.rows(), vert.cols());
    grad.resize(2, 3);

    grad.col(0) = d2 * e2 - d3 * e3 + f1 * rotate_90deg(e1);
    grad.col(1) = d3 * e3 - d1 * e1 + f1 * rotate_90deg(e2);
    grad.col(2) = d1 * e1 - d2 * e2 + f1 * rotate_90deg(e3);

    grad /= energy;
    return energy;
}

double
TLC_2D_Formulation::compute_lifted_TriArea_with_gradient_projected_residual_Hessian(const Eigen::Vector2d &v1,
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
    double energy = sqrt(signed_area * signed_area + d1 * e1.squaredNorm()
                         + d2 * e2.squaredNorm() + d3 * e3.squaredNorm()
                         + scaled_squared_rest_area);

    double f1 = signed_area / 2;

//    grad.resize(vert.rows(), vert.cols());
    grad.resize(2, 3);

    grad.col(0) = d2 * e2 - d3 * e3 + f1 * rotate_90deg(e1);
    grad.col(1) = d3 * e3 - d1 * e1 + f1 * rotate_90deg(e2);
    grad.col(2) = d1 * e1 - d2 * e2 + f1 * rotate_90deg(e3);

    grad /= energy;

    // PSD projected Hessian of (generalized triangle area - signed triangle area)
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
    double eigen_value_twist = (alpha + I3) / energy_density - 1;
    Eigen::Vector4d eigen_vec_twist;
    vectorize2x2((U.col(1) * V.col(0).transpose() - U.col(0) * V.col(1).transpose()) / sqrt(2.),
                 eigen_vec_twist);
    //// flip
    double eigen_value_flip = (alpha - I3) / energy_density + 1;
    Eigen::Vector4d eigen_vec_flip;
    vectorize2x2((U.col(1) * V.col(0).transpose() + U.col(0) * V.col(1).transpose()) / sqrt(2.),
                 eigen_vec_flip);
    //// scale
    Eigen::Vector4d vec_d1, vec_d2;
    vectorize2x2(U.col(0) * V.col(0).transpose(), vec_d1);
    vectorize2x2(U.col(1) * V.col(1).transpose(), vec_d2);

    double energy_cube = energy_density * energy_density * energy_density;
    double a11 = alpha * (alpha + s2 * s2) * (alpha + s2 * s2) / energy_cube;
    double a22 = alpha * (alpha + s1 * s1) * (alpha + s1 * s1) / energy_cube;
    double a12 = I3 * (I3 * I3 + alpha * (I2 + alpha)) / energy_cube - 1;
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
