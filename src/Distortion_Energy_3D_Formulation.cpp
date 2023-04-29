//
// Created by Charles Du on 4/28/23.
//

#include "Distortion_Energy_3D_Formulation.h"
#include "deformation_gradient_util.h"
#include "geo_util.h"
#include <Eigen/Eigenvalues>

Distortion_Energy_3D_Formulation::Distortion_Energy_3D_Formulation(const Eigen::Matrix3Xd& restV_, Eigen::Matrix3Xd initV,
                                                                   Eigen::Matrix4Xi T_,
                                                                   const Eigen::VectorXi &handles)
        : Injective_Energy_3D_Formulation(restV_, std::move(initV), std::move(T_), handles) {
    // compute rest tetrahedron volumes
    rest_tetVolumes.resize(T.cols());
    for (int i = 0; i < T.cols(); ++i) {
        rest_tetVolumes(i) = abs(compute_tet_signed_volume(restV.col(T(0, i)),
                                                           restV.col(T(1, i)),
                                                           restV.col(T(2, i)),
                                                           restV.col(T(3, i))));
    }

    // compute inverse of edge matrices
    rest_invEdgeMat.reserve(T.cols());
    Eigen::Matrix3d rest_edge_mat;
    for (int i = 0; i < T.cols(); ++i) {
        rest_invEdgeMat.emplace_back();
        compute_edge_matrix(restV.col(T(0, i)),
                            restV.col(T(1, i)),
                            restV.col(T(2, i)),
                            restV.col(T(3, i)),
                            rest_edge_mat);
        rest_invEdgeMat.back() = rest_edge_mat.inverse();
    }

    // compute derivative of deformation gradient w.r.t. target vertices
    pFpx_list.reserve(T.cols());
    for (int i = 0; i < T.cols(); ++i) {
        pFpx_list.emplace_back();
        compute_flatten_pFpx(rest_invEdgeMat[i], pFpx_list.back());
    }

}

double Distortion_Energy_3D_Formulation::compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) {
    update_x_and_V(x);

    energy_list.resize(free_tetI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_tetI.size(); ++i) {
        auto ti = free_tetI(i);

        energy_list(i) = compute_tetrahedron_energy(V.col(T(0, ti)),
                                                    V.col(T(1, ti)),
                                                    V.col(T(2, ti)),
                                                    V.col(T(3, ti)),
                                                    rest_tetVolumes(ti),
                                                    rest_invEdgeMat[ti]);
    }

    return energy_list.sum();
}

double Distortion_Energy_3D_Formulation::compute_tetrahedron_energy(const Eigen::Vector3d &v1,
                                                                    const Eigen::Vector3d &v2,
                                                                    const Eigen::Vector3d &v3,
                                                                    const Eigen::Vector3d &v4, double rest_tetVolume,
                                                                    const Eigen::Matrix3d &rest_inv_EdgeMat) {
    // compute deformation gradient
    Eigen::Matrix3d target_edge_mat;
    compute_edge_matrix(v1, v2, v3, v4, target_edge_mat);
    Eigen::Matrix3d deformation_gradient = target_edge_mat * rest_inv_EdgeMat;
    // compute SVD
    Eigen::Matrix3d U, V;
    Eigen::Vector3d singular_values;
    compute_SVD(deformation_gradient, U, singular_values, V);
    double s1 = singular_values[0];
    double s2 = singular_values[1];
    double s3 = singular_values[2];
    // S-centric invariants
    double I1 = s1 + s2 + s3;
    double I2 = s1*s1 + s2*s2 + s3*s3;
    double I3 = s1*s2*s3;
    // compute energy
    return rest_tetVolume * compute_psi(I1, I2, I3);
}


double Distortion_Energy_3D_Formulation::compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) {
    update_x_and_V(x);

    Eigen::VectorXd energyList(free_tetI.size());
    std::vector<Eigen::Matrix3Xd> gradList(free_tetI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_tetI.size(); ++i) {
        auto ti = free_tetI(i);

        energyList(i) = compute_tetrahedron_energy_with_gradient(V.col(T(0, ti)),
                                                                  V.col(T(1, ti)),
                                                                  V.col(T(2, ti)),
                                                                  V.col(T(3, ti)),
                                                                  rest_tetVolumes(ti),
                                                                  rest_invEdgeMat[ti],
                                                                  pFpx_list[ti],
                                                                  gradList[i]);
    }

    // accumulate gradient
    Eigen::Matrix3Xd gradV = Eigen::Matrix3Xd::Zero(3, V.cols());
    for (auto i = 0; i < free_tetI.size(); ++i) {
        auto ti = free_tetI(i);
        gradV.col(T(0, ti)) += gradList[i].col(0);
        gradV.col(T(1, ti)) += gradList[i].col(1);
        gradV.col(T(2, ti)) += gradList[i].col(2);
        gradV.col(T(3, ti)) += gradList[i].col(3);
    }

    // flatten gradient of free vertices
    grad.resize(freeI.size() * 3);
    for (auto i = 0; i < freeI.size(); ++i) {
        grad.segment<3>(i * 3) = gradV.col(freeI(i));
    }

    return energyList.sum();
}

double Distortion_Energy_3D_Formulation::compute_tetrahedron_energy_with_gradient(const Eigen::Vector3d &v1,
                                                                                  const Eigen::Vector3d &v2,
                                                                                  const Eigen::Vector3d &v3,
                                                                                  const Eigen::Vector3d &v4,
                                                                                  double rest_tetVolume,
                                                                                  const Eigen::Matrix3d &rest_inv_EdgeMat,
                                                                                  const Eigen::MatrixXd &pFpx,
                                                                                  Eigen::Matrix3Xd &grad) {
    // compute deformation gradient
    Eigen::Matrix3d target_edge_mat;
    compute_edge_matrix(v1, v2, v3, v4, target_edge_mat);
    Eigen::Matrix3d deformation_gradient = target_edge_mat * rest_inv_EdgeMat;
    // compute SVD
    Eigen::Matrix3d U, V;
    Eigen::Vector3d singular_values;
    compute_SVD(deformation_gradient, U, singular_values, V);
    double s1 = singular_values[0];
    double s2 = singular_values[1];
    double s3 = singular_values[2];
    // S-centric invariants
    double I1 = s1 + s2 + s3;
    double I2 = s1*s1 + s2*s2 + s3*s3;
    double I3 = s1*s2*s3;
    // compute psi and its gradient w.r.t. I1, I2, I3
    Eigen::Vector3d grad_psi;
    double psi = compute_psi_with_gradient(I1, I2, I3, grad_psi);
    // compute gradient of psi w.r.t. flattened deformation gradient
    Vector9d f;
    vectorize3x3(deformation_gradient, f);
    Vector9d d_I1_d_f;
    compute_d_I1_d_f(U, V, d_I1_d_f);
    Vector9d d_I2_d_f = compute_d_I2_d_f(f);
    Vector9d d_I3_d_f;
    compute_d_I3_d_f(U, singular_values, V, d_I3_d_f);
    Vector9d d_psi_df = grad_psi.x() * d_I1_d_f + grad_psi.y() * d_I2_d_f + grad_psi.z() * d_I3_d_f;
    // compute gradient of energy
    Eigen::Vector<double, 12> d_psi_d_x = pFpx.transpose() * d_psi_df;
    grad.resize(3, 4);
    grad.col(0) = d_psi_d_x.segment<3>(0);
    grad.col(1) = d_psi_d_x.segment<3>(3);
    grad.col(2) = d_psi_d_x.segment<3>(6);
    grad.col(3) = d_psi_d_x.segment<3>(9);
    grad *= rest_tetVolume;
    // return energy
    return rest_tetVolume * psi;
}

double Distortion_Energy_3D_Formulation::compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x,
                                                                              Eigen::VectorXd &energy_list,
                                                                              Eigen::VectorXd &grad, SpMat &Hess) {
    update_x_and_V(x);

    energy_list.resize(free_tetI.size());
    std::vector<Eigen::Matrix3Xd> gradList(free_tetI.size());

    const int vDim = 3;
    const int simplex_size = 4;
    std::vector<Eigen::Triplet<double>> tripletList(simplex_size * simplex_size * vDim * vDim * free_tetI.size());

#pragma omp parallel
#pragma omp for
    for (auto i = 0; i < free_tetI.size(); ++i) {
        auto ti = free_tetI(i);
        Eigen::MatrixXd hess;
        energy_list(i) = compute_tetrahedron_energy_with_gradient_projected_Hessian(V.col(T(0, ti)),
                                                                           V.col(T(1, ti)),
                                                                           V.col(T(2, ti)),
                                                                           V.col(T(3, ti)),
                                                                           rest_tetVolumes(ti),
                                                                           rest_invEdgeMat[ti],
                                                                           pFpx_list[ti],
                                                                           gradList[i],
                                                                           hess);

        // update Hessian of free vertices
        int current_index = i * simplex_size * simplex_size * vDim * vDim;
        Eigen::Vector4i indices = T_free.col(ti);
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
    Eigen::Matrix3Xd gradV = Eigen::Matrix3Xd::Zero(3, V.cols());
    for (auto i = 0; i < free_tetI.size(); ++i) {
        auto ti = free_tetI(i);
        gradV.col(T(0, ti)) += gradList[i].col(0);
        gradV.col(T(1, ti)) += gradList[i].col(1);
        gradV.col(T(2, ti)) += gradList[i].col(2);
        gradV.col(T(3, ti)) += gradList[i].col(3);
    }

    // flatten gradient of free vertices
    grad.resize(freeI.size() * 3);
    for (auto i = 0; i < freeI.size(); ++i) {
        grad.segment<3>(i * 3) = gradV.col(freeI(i));
    }

    return energy_list.sum();
}

double Distortion_Energy_3D_Formulation::compute_tetrahedron_energy_with_gradient_projected_Hessian(
        const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v3, const Eigen::Vector3d &v4,
        double rest_tetVolume, const Eigen::Matrix3d &rest_inv_EdgeMat, const Eigen::MatrixXd &pFpx,
        Eigen::Matrix3Xd &grad, Eigen::MatrixXd &hess) {
    // compute deformation gradient
    Eigen::Matrix3d target_edge_mat;
    compute_edge_matrix(v1, v2, v3, v4, target_edge_mat);
    Eigen::Matrix3d deformation_gradient = target_edge_mat * rest_inv_EdgeMat;
    // compute SVD
    Eigen::Matrix3d U, V;
    Eigen::Vector3d singular_values;
    compute_SVD(deformation_gradient, U, singular_values, V);
    double s1 = singular_values[0];
    double s2 = singular_values[1];
    double s3 = singular_values[2];
    // S-centric invariants
    double I1 = s1 + s2 + s3;
    double I2 = s1*s1 + s2*s2 + s3*s3;
    double I3 = s1*s2*s3;
    // compute psi and its gradient w.r.t. I1, I2, I3
    Eigen::Vector3d grad_psi;
    double psi = compute_psi_with_gradient(I1, I2, I3, grad_psi);
    // compute gradient of psi w.r.t. flattened deformation gradient
    Vector9d f;
    vectorize3x3(deformation_gradient, f);
    Vector9d d_I1_d_f;
    compute_d_I1_d_f(U, V, d_I1_d_f);
    Vector9d d_I2_d_f = compute_d_I2_d_f(f);
    Vector9d d_I3_d_f;
    compute_d_I3_d_f(U, singular_values, V, d_I3_d_f);
    Vector9d d_psi_df = grad_psi.x() * d_I1_d_f + grad_psi.y() * d_I2_d_f + grad_psi.z() * d_I3_d_f;
    // compute gradient of energy
    Eigen::Vector<double, 12> d_psi_d_x = pFpx.transpose() * d_psi_df;
    grad.resize(3, 4);
    grad.col(0) = d_psi_d_x.segment<3>(0);
    grad.col(1) = d_psi_d_x.segment<3>(3);
    grad.col(2) = d_psi_d_x.segment<3>(6);
    grad.col(3) = d_psi_d_x.segment<3>(9);
    grad *= rest_tetVolume;
    // compute Hessian
    Vector9d lambdas;
    Eigen::Matrix3d matA;
    bool scale_decoupled = compute_analytic_eigen_information(s1, s2, s3, I1, I2, I3, lambdas, matA);
    Eigen::Matrix<double,9,9> f_Hess;
    f_Hess.setZero();
    Vector9d eigen_vec;
    //// twist 1
    double eigen_value_twist_1 = lambdas[3];
    if (eigen_value_twist_1 > 0) {
//        Vector9d eigen_vec_twist_1;
        // 0 0 0
        // 0 0 1
        // 0 -1 0
        vectorize3x3((U.col(1) * V.col(2).transpose() - U.col(2) * V.col(1).transpose()) / sqrt(2.), eigen_vec);
        f_Hess += eigen_value_twist_1 * eigen_vec * eigen_vec.transpose();
    }
    //// twist 2
    double eigen_value_twist_2 = lambdas[4];
    if (eigen_value_twist_2 > 0) {
//        Vector9d eigen_vec_twist_2;
        // 0 0 1
        // 0 0 0
        // -1 0 0
        vectorize3x3((U.col(0) * V.col(2).transpose() - U.col(2) * V.col(0).transpose()) / sqrt(2.), eigen_vec);
        f_Hess += eigen_value_twist_2 * eigen_vec * eigen_vec.transpose();
    }
    //// twist 3
    double eigen_value_twist_3 = lambdas[5];
    if (eigen_value_twist_3 > 0) {
//        Vector9d eigen_vec_twist_3;
        // 0 -1 0
        // 1 0 0
        // 0 0 0
        vectorize3x3((U.col(1) * V.col(0).transpose() - U.col(0) * V.col(1).transpose()) / sqrt(2.), eigen_vec);
        f_Hess += eigen_value_twist_3 * eigen_vec * eigen_vec.transpose();
    }
    //// flip 1
    double eigen_value_flip_1 = lambdas[6];
    if (eigen_value_flip_1 > 0) {
//        Vector9d eigen_vec_flip_1;
        // 0 0 0
        // 0 0 1
        // 0 1 0
        vectorize3x3((U.col(1) * V.col(2).transpose() + U.col(2) * V.col(1).transpose()) / sqrt(2.), eigen_vec);
        f_Hess += eigen_value_flip_1 * eigen_vec * eigen_vec.transpose();
    }
    //// flip 2
    double eigen_value_flip_2 = lambdas[7];
    if (eigen_value_flip_2 > 0) {
//        Vector9d eigen_vec_flip_2;
        // 0 0 1
        // 0 0 0
        // 1 0 0
        vectorize3x3((U.col(0) * V.col(2).transpose() + U.col(2) * V.col(0).transpose()) / sqrt(2.), eigen_vec);
        f_Hess += eigen_value_flip_2 * eigen_vec * eigen_vec.transpose();
    }
    //// flip 3
    double eigen_value_flip_3 = lambdas[8];
    if (eigen_value_flip_3 > 0) {
//        Vector9d eigen_vec_flip_3;
        // 0 1 0
        // 1 0 0
        // 0 0 0
        vectorize3x3((U.col(0) * V.col(1).transpose() + U.col(1) * V.col(0).transpose()) / sqrt(2.), eigen_vec);
        f_Hess += eigen_value_flip_3 * eigen_vec * eigen_vec.transpose();
    }
    //// scale
    if (scale_decoupled) {
        if (lambdas[0] > 0) {
            vectorize3x3(U.col(0) * V.col(0).transpose(), eigen_vec);
            f_Hess += lambdas[0] * eigen_vec * eigen_vec.transpose();
        }
        if (lambdas[1] > 0) {
            vectorize3x3(U.col(1) * V.col(1).transpose(), eigen_vec);
            f_Hess += lambdas[1] * eigen_vec * eigen_vec.transpose();
        }
        if (lambdas[2] > 0) {
            vectorize3x3(U.col(2) * V.col(2).transpose(), eigen_vec);
            f_Hess += lambdas[2] * eigen_vec * eigen_vec.transpose();
        }
    } else {
        Vector9d vec_d1, vec_d2, vec_d3;
        vectorize3x3(U.col(0) * V.col(0).transpose(), vec_d1);
        vectorize3x3(U.col(1) * V.col(1).transpose(), vec_d2);
        vectorize3x3(U.col(2) * V.col(2).transpose(), vec_d3);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(matA);
        Eigen::Vector3d matA_eigenVals = eigenSolver.eigenvalues();
        // eigen vectors has been normalized to have length 1
        Eigen::Matrix3d matA_eigenVecs = eigenSolver.eigenvectors();
        //
        if (matA_eigenVals[0] > 0) {
            eigen_vec = matA_eigenVecs(0,0) * vec_d1 + matA_eigenVecs(1,0) * vec_d2 + matA_eigenVecs(2,0) * vec_d3;
            f_Hess += matA_eigenVals[0] * eigen_vec * eigen_vec.transpose();
        }
        if (matA_eigenVals[1] > 0) {
            eigen_vec = matA_eigenVecs(0,1) * vec_d1 + matA_eigenVecs(1,1) * vec_d2 + matA_eigenVecs(2,1) * vec_d3;
            f_Hess += matA_eigenVals[1] * eigen_vec * eigen_vec.transpose();
        }
        if (matA_eigenVals[2] > 0) {
            eigen_vec = matA_eigenVecs(0,2) * vec_d1 + matA_eigenVecs(1,2) * vec_d2 + matA_eigenVecs(2,2) * vec_d3;
            f_Hess += matA_eigenVals[2] * eigen_vec * eigen_vec.transpose();
        }
    }
    // PSD x-Hessian
    hess = pFpx.transpose() * (rest_tetVolume * f_Hess) * pFpx;
    // return energy
    return rest_tetVolume * psi;
}