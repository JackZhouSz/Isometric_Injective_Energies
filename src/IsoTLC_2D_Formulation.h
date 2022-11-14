//
// Created by Charles Du on 11/13/22.
//

#ifndef ISO_TLC_SEA_ISOTLC_2D_FORMULATION_H
#define ISO_TLC_SEA_ISOTLC_2D_FORMULATION_H

#include "TLC_2D_Formulation.h"

class IsoTLC_2D_Formulation : public TLC_2D_Formulation {
public:
    IsoTLC_2D_Formulation(const Eigen::MatrixXd &restV, Eigen::Matrix2Xd initV,
                          Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles,
                          std::string form_, double alpha_);

    ~IsoTLC_2D_Formulation() override = default;

protected:
    // compute isometric lifted triangle area
    // input:
    //  - v1, v2, v3: three vertices
    //  - r: squared edge lengths of aux triangle
    double compute_lifted_TriArea(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2, const Eigen::Vector2d &v3,
                                  const Eigen::Vector3d &Dirichlet_coefficient, double scaled_squared_rest_area,
                                  double &signed_area) const override;

    // compute residual (isometric lifted area - signed area)
    // input:
    //  - v1, v2, v3: three vertices
    //  - r: squared edge lengths of aux triangle
    double compute_residual(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2, const Eigen::Vector2d &v3,
                            const Eigen::Vector3d &Dirichlet_coefficient, double scaled_squared_rest_area,
                            double &signed_area) const override;

    // compute isometric lifted triangle area with gradient wrt. vert
    // input:
    //  - v1, v2, v3: three vertices
    //  - r: squared edge lengths of aux triangle
    double compute_lifted_TriArea_with_gradient(const Eigen::Vector2d &v1, const Eigen::Vector2d &v2,
                                                const Eigen::Vector2d &v3, const Eigen::Vector3d &Dirichlet_coefficient,
                                                double scaled_squared_rest_area, Eigen::Matrix2Xd &grad,
                                                double &signed_area) const override;

    // compute isometric lifted triangle area with gradient and PSD-projected Hessian of residual wrt. vert
    // input:
    //  - vert: three vertices
    //  - r: squared edge lengths of aux triangle
    double compute_lifted_TriArea_with_gradient_projected_residual_Hessian(const Eigen::Vector2d &v1,
                                                                           const Eigen::Vector2d &v2,
                                                                           const Eigen::Vector2d &v3,
                                                                           const Eigen::Vector3d &Dirichlet_coefficient,
                                                                           double scaled_squared_rest_area,
                                                                           double rest_area,
                                                                           const Eigen::Matrix2d &rest_inverse_EdgeMat,
                                                                           const Eigen::MatrixXd &pFpx,
                                                                           Eigen::Matrix2Xd &grad,
                                                                           Eigen::MatrixXd &Hess,
                                                                           double &signed_area) const override;

private:
    // coefficient for the squared areas of target triangles: (1+alpha/2)
    double squared_targetA_scale_coeff;

    // coefficients for analytic eigen-system of Hessian
    double half_alpha;
    double one_plus_half_alpha;
    double coeff_diag;
    double coeff_off_diag;
};


#endif //ISO_TLC_SEA_ISOTLC_2D_FORMULATION_H
