//
// Created by Charles Du on 4/29/23.
//

#ifndef ISO_TLC_SEA_ISOTLC_RESIDUAL_2D_FORMULATION_H
#define ISO_TLC_SEA_ISOTLC_RESIDUAL_2D_FORMULATION_H

#include "TLC_Residual_2D_Formulation.h"

#include <utility>

class IsoTLC_Residual_2D_Formulation : public TLC_Residual_2D_Formulation {
public:
    IsoTLC_Residual_2D_Formulation(const Eigen::MatrixXd& restV_, Eigen::Matrix2Xd initV,
                                   Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles,
                                   std::string form, double alpha)
            : TLC_Residual_2D_Formulation(restV_, std::move(initV), std::move(F_), handles, std::move(form), alpha) {};

    ~IsoTLC_Residual_2D_Formulation() override = default;

protected:
    double compute_psi(double I1, double I2, double I3) override {
        return sqrt(alpha * (0.5 + alpha) + (alpha*I2)/2. + (1 + alpha/2.) * I3 * I3) - I3;
    }

    double compute_psi_with_gradient(double I1, double I2, double I3, Eigen::Vector3d &grad_psi) override {
        double psi_lc = sqrt(alpha * (0.5 + alpha) + (alpha*I2)/2. + (1 + alpha/2.) * I3 * I3);
        grad_psi << 0, 0.25 * alpha / psi_lc, (1 + alpha/2.) * I3 / psi_lc - 1;
        return psi_lc - I3;
    }

    bool compute_analytic_eigen_information(double sigma1, double sigma2,
                                            double I1, double I2, double I3,
                                            Eigen::Vector4d &lambdas, Eigen::Matrix2d &matA) override {

        // Precompute common sub-expressions
        double sigma1_sq = pow(sigma1, 2);
        double sigma2_sq = pow(sigma2, 2);
        double sigma1_pow3 = pow(sigma1, 3);
        double sigma2_pow3 = pow(sigma2, 3);
        double alpha_sq = pow(alpha, 2);
        double sigma1_pow4 = pow(sigma1, 4);
        double sigma2_pow4 = pow(sigma2, 4);
        double I3_sq = pow(I3, 2);
        double I3_pow3 = pow(I3, 3);
        double psi_lc = sqrt(alpha_sq + I3_sq + (alpha * (1 + sigma1_sq) * (1 + sigma2_sq)) / 2.);

        // Compute lambda values
        lambdas(0) = (alpha * (2 * alpha_sq * (1 + sigma2_sq) + 2 * (sigma2_sq + sigma2_pow4) +
                               alpha * (1 + 6 * sigma2_sq + sigma2_pow4))) /
                     (4 * pow(psi_lc, 3));

        lambdas(1) = (alpha * (2 * alpha_sq * (1 + sigma1_sq) + 2 * (sigma1_sq + sigma1_pow4) +
                               alpha * (1 + 6 * sigma1_sq + sigma1_pow4))) /
                     (4 * pow(psi_lc, 3));

        lambdas(2) = I3 / psi_lc - 1 + (alpha * (1 + I3)) / (2 * psi_lc);

        lambdas(3) = (alpha - (2 + alpha) * I3) / (2 * psi_lc) + 1;

        // Compute off-diagonal value
        double off_diagonal = (4 * alpha * I3 + 9 * alpha_sq * I3 + 4 * alpha_sq * alpha * I3 +
                               2 * alpha * sigma1_pow3 * sigma2 + alpha_sq * sigma1_pow3 * sigma2 +
                               2 * alpha * sigma1 * sigma2_pow3 + alpha_sq * sigma1 * sigma2_pow3 +
                               4 * I3_pow3 + 4 * alpha * I3_pow3 + alpha_sq * I3_pow3) /
                              (4. * pow(psi_lc, 3)) - 1;


        // Fill matA with lambda values and off-diagonal value
        matA << lambdas(0), off_diagonal,
                off_diagonal, lambdas(1);

        // Return true if matA is diagonal
        return off_diagonal == 0;
    }

    };


#endif //ISO_TLC_SEA_ISOTLC_RESIDUAL_2D_FORMULATION_H
