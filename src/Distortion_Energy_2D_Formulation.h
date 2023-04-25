//
// Created by Charles Du on 4/24/23.
//

#ifndef ISO_TLC_SEA_DISTORTION_ENERGY_2D_FORMULATION_H
#define ISO_TLC_SEA_DISTORTION_ENERGY_2D_FORMULATION_H

#include "Injective_Energy_2D_Formulation.h"

class Distortion_Energy_2D_Formulation : public Injective_Energy_2D_Formulation {
public:
    Distortion_Energy_2D_Formulation(Eigen::MatrixXd restV_, Eigen::Matrix2Xd initV,
                                     Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles);

    ~Distortion_Energy_2D_Formulation() override = default;

    double compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) override;

    double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override;

    double compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list,
                                                Eigen::VectorXd &grad, SpMat &Hess) override;

protected:
    // areas of rest triangles
    Eigen::VectorXd restA;
    // inverse of edge matrices of rest triangles
    std::vector<Eigen::Matrix2d> rest_invEdgeMat;
    // pFpx: derivative of deformation gradient w.r.t. vertices coordinates
    std::vector<Eigen::MatrixXd> pFpx_list;

    double compute_triangle_energy(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2, const Eigen::Vector2d &v3,
                                   double rest_triangle_area, const Eigen::Matrix2d &rest_inv_EdgeMat);

    double compute_triangle_energy_with_gradient(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2,
                                                 const Eigen::Vector2d &v3, double rest_triangle_area,
                                                 const Eigen::Matrix2d &rest_inv_EdgeMat,
                                                 const Eigen::MatrixXd &pFpx,
                                                 Eigen::Matrix2Xd &grad);

    double compute_triangle_energy_with_gradient_projected_Hessian(const Eigen::Vector2d &v1,
                                                                   const Eigen::Vector2d &v2,
                                                                   const Eigen::Vector2d &v3,
                                                                   double rest_triangle_area,
                                                                   const Eigen::Matrix2d &rest_inv_EdgeMat,
                                                                   const Eigen::MatrixXd &pFpx,
                                                                   Eigen::Matrix2Xd &grad,
                                                                   Eigen::MatrixXd &hess);

    virtual double compute_psi(double I1, double I2, double I3) = 0;

    virtual double compute_psi_with_gradient(double I1, double I2, double I3, Eigen::Vector3d &grad_psi) = 0;

    virtual bool compute_analytic_eigen_information(double sigma1, double sigma2,
                                                    double I1, double I2, double I3,
                                                    Eigen::Vector4d &lambdas, Eigen::Matrix2d &matA) = 0;
};


#endif //ISO_TLC_SEA_DISTORTION_ENERGY_2D_FORMULATION_H
