//
// Created by Charles Du on 4/28/23.
//

#ifndef ISO_TLC_SEA_DISTORTION_ENERGY_3D_FORMULATION_H
#define ISO_TLC_SEA_DISTORTION_ENERGY_3D_FORMULATION_H

#include "Injective_Energy_3D_Formulation.h"

class Distortion_Energy_3D_Formulation : public Injective_Energy_3D_Formulation {
public:
    Distortion_Energy_3D_Formulation(const Eigen::Matrix3Xd& restV_, Eigen::Matrix3Xd initV,
                                     Eigen::Matrix4Xi T_, const Eigen::VectorXi &handles);

    ~Distortion_Energy_3D_Formulation() override = default;

    double compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) override;

    double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override;

    double compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list,
                                                Eigen::VectorXd &grad, SpMat &Hess) override;

protected:
    // volumes of rest tetrahedrons
    Eigen::VectorXd rest_tetVolumes;
    // inverse of edge matrices of rest tetrahedrons
    std::vector<Eigen::Matrix3d> rest_invEdgeMat;
    // pFpx: derivative of deformation gradient w.r.t. vertices coordinates
    std::vector<Eigen::MatrixXd> pFpx_list;

    double compute_tetrahedron_energy(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v3,
                                      const Eigen::Vector3d &v4, double rest_tetVolume, const Eigen::Matrix3d &rest_inv_EdgeMat);

    double compute_tetrahedron_energy_with_gradient(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2,
                                                    const Eigen::Vector3d &v3, const Eigen::Vector3d &v4,
                                                    double rest_tetVolume, const Eigen::Matrix3d &rest_inv_EdgeMat,
                                                    const Eigen::MatrixXd &pFpx,
                                                    Eigen::Matrix3Xd &grad);

    double compute_tetrahedron_energy_with_gradient_projected_Hessian(const Eigen::Vector3d &v1,
                                                                      const Eigen::Vector3d &v2,
                                                                      const Eigen::Vector3d &v3,
                                                                      const Eigen::Vector3d &v4,
                                                                      double rest_tetVolume,
                                                                      const Eigen::Matrix3d &rest_inv_EdgeMat,
                                                                      const Eigen::MatrixXd &pFpx,
                                                                      Eigen::Matrix3Xd &grad,
                                                                      Eigen::MatrixXd &hess);
    // compute psi from invariant I1, I2, I3
    // such that the distortion of a tetrahedron is rest_tetVolume * psi
    virtual double compute_psi(double I1, double I2, double I3) = 0;

    // compute psi from invariant I1, I2, I3
    // also compute the gradient of psi w.r.t. I1, I2, I3
    virtual double compute_psi_with_gradient(double I1, double I2, double I3, Eigen::Vector3d &grad_psi) = 0;

    // compute the analytic eigen-system of the Hessian of psi w.r.t. flattened deformation gradient
    // input:
    // - singular values sigma1, sigma2, sigma3
    // - invariants I1, I2, I3
    // output:
    // - eigenvalues (lambdas): scale1, scale2, scale3, twist1, twist2, twist3, flip1, flip2, flip3
    // - stretching matrix: matA
    // return: true if the scaling modes are decoupled, i.e. matA is diagonal
    virtual bool compute_analytic_eigen_information(double sigma1, double sigma2, double sigma3,
                                                      double I1, double I2, double I3,
                                                      Eigen::Vector<double,9> &lambdas, Eigen::Matrix3d &V) = 0;
};


#endif //ISO_TLC_SEA_DISTORTION_ENERGY_3D_FORMULATION_H
