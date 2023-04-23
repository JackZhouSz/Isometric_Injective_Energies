//
// Created by Charles Du on 11/12/22.
//

#ifndef ISO_TLC_SEA_TLC_2D_FORMULATION_H
#define ISO_TLC_SEA_TLC_2D_FORMULATION_H

#include "Injective_Energy_2D_Formulation.h"

class TLC_2D_Formulation : public Injective_Energy_2D_Formulation {
public:
    TLC_2D_Formulation(const Eigen::MatrixXd &restV, Eigen::Matrix2Xd initV,
                       Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles,
                       std::string form_, double alpha_);

    ~TLC_2D_Formulation() override = default;

    double compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) override;

    double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override;

    double
    compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list, Eigen::VectorXd &grad,
                                         SpMat &Hess) override;

    bool is_injective() override;

protected:
    std::string form = "harmonic";
    double alpha = 1e-4;

    // get ready for computing energy and derivatives
    void initialize();

    // squared edge lengths of auxiliary triangles
    Eigen::Matrix3Xd restD;
    // areas of auxiliary triangles
    Eigen::VectorXd restA;
    // inverse of edge matrices of auxiliary triangles
    std::vector<Eigen::Matrix2d> rest_invEdgeMat;
    // pFpx: derivative of deformation gradient w.r.t. vertices coordinates
    std::vector<Eigen::MatrixXd> pFpx_list;

    // scaled squared areas of auxiliary triangles: alpha^2*A_rest^2
    Eigen::VectorXd scaled_squared_restA;
    // scaled Dirichlet coefficients: (alpha/8)(d_{i+1} + d{i+2} - d_i)
    // where d_i is the squared length of i-th edge of the auxiliary triangle
    Eigen::Matrix3Xd scaled_Dirichlet_coefficients;

    // compute total lifted content, record lifted content of each triangle
    double compute_total_lifted_content(const Eigen::Matrix2Xd &vertices,
                                        Eigen::VectorXd &energyList);

    // compute total residual, record (lifted content - signed content) of each triangle
    double compute_total_residual(const Eigen::Matrix2Xd &vertices,
                                  Eigen::VectorXd &residual_list);

    // compute total lifted content and its gradient
    double compute_total_lifted_content_with_gradient(const Eigen::Matrix2Xd &vertices,
                                                      Eigen::Matrix2Xd &grad);

    // compute total lifted content and its gradient,
    // and PSD projected Hessian of (TLC - total signed area) on free vertices
    double compute_total_lifted_content_with_gradient_and_projectedHessian(
            const Eigen::Matrix2Xd &vertices,
            const Eigen::VectorXi &freeI,
            const Eigen::Matrix3Xi &F_free,
            // output
            Eigen::VectorXd &lifted_content_list,
            Eigen::Matrix2Xd &grad,
            SpMat &Hess);

protected:
    // compute lifted triangle area
    // input:
    //  - v1, v2, v3: three vertices
    virtual double compute_lifted_TriArea(
            const Eigen::Vector2d &v1,
            const Eigen::Vector2d &v2,
            const Eigen::Vector2d &v3,
            const Eigen::Vector3d &Dirichlet_coefficient,
            double scaled_squared_rest_area,
            double &signed_area) const;

    // compute residual (lifted area - signed area)
    // input:
    //  - v1, v2, v3: three vertices
    virtual double compute_residual(
            const Eigen::Vector2d &v1,
            const Eigen::Vector2d &v2,
            const Eigen::Vector2d &v3,
            const Eigen::Vector3d &Dirichlet_coefficient,
            double scaled_squared_rest_area,
            double &signed_area) const;

    // compute lifted triangle area with gradient wrt. vert
    // input:
    //  - v1, v2, v3: three vertices
    virtual double compute_lifted_TriArea_with_gradient(
            const Eigen::Vector2d &v1,
            const Eigen::Vector2d &v2,
            const Eigen::Vector2d &v3,
            const Eigen::Vector3d &Dirichlet_coefficient,
            double scaled_squared_rest_area,
            Eigen::Matrix2Xd &grad,
            double &signed_area) const;

    // compute lifted triangle area with gradient and PSD-projected Hessian of residual wrt. vert
    // input:
    //  - vert: three vertices
    virtual double compute_lifted_TriArea_with_gradient_projected_residual_Hessian(
            const Eigen::Vector2d &v1, const Eigen::Vector2d &v2, const Eigen::Vector2d &v3,
            const Eigen::Vector3d &Dirichlet_coefficient,
            double scaled_squared_rest_area,
            double rest_area,
            const Eigen::Matrix2d &rest_inverse_EdgeMat,
            const Eigen::MatrixXd &pFpx,
            Eigen::Matrix2Xd &grad, Eigen::MatrixXd &Hess,
            double &signed_area) const;
};


#endif //ISO_TLC_SEA_TLC_2D_FORMULATION_H
