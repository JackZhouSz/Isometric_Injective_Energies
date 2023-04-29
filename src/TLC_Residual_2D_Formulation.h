//
// Created by Charles Du on 4/29/23.
//

#ifndef ISO_TLC_SEA_TLC_RESIDUAL_2D_FORMULATION_H
#define ISO_TLC_SEA_TLC_RESIDUAL_2D_FORMULATION_H

#include "Distortion_Energy_2D_Formulation.h"

class TLC_Residual_2D_Formulation : public Distortion_Energy_2D_Formulation {
public:
    TLC_Residual_2D_Formulation(const Eigen::MatrixXd &restV, Eigen::Matrix2Xd initV,
                                Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles,
                                std::string form_, double alpha_);

    ~TLC_Residual_2D_Formulation() override = default;

protected:
    std::string form = "harmonic";
    double alpha = 1e-4;

protected:
    double compute_psi(double I1, double I2, double I3) override {
        return sqrt(I3 * I3 + alpha * I2 + alpha * alpha) - I3;
    }

    double compute_psi_with_gradient(double I1, double I2, double I3, Eigen::Vector3d &grad_psi) override {
        double psi_lc = sqrt(I3 * I3 + alpha * I2 + alpha * alpha);
        grad_psi << 0, 0.5 * alpha / psi_lc, I3 / psi_lc - 1;
        return psi_lc - I3;
    }

    bool compute_analytic_eigen_information(double sigma1, double sigma2,
                                            double I1, double I2, double I3,
                                            Eigen::Vector4d &lambdas, Eigen::Matrix2d &matA) override {
        double alpha_sq_sigma1 = pow(alpha + pow(sigma1, 2), 2);
        double alpha_sq_sigma2 = pow(alpha + pow(sigma2, 2), 2);
        double psi_lc = sqrt(I3 * I3 + alpha * I2 + alpha * alpha);
        double denominator = pow(psi_lc, 3);

        lambdas << (alpha * alpha_sq_sigma2) / denominator,
                (alpha * alpha_sq_sigma1) / denominator,
                (alpha + I3) / psi_lc - 1,
                (alpha - I3) / psi_lc + 1;

        double off_diagonal = I3 / psi_lc - 1;
        matA << lambdas(0), off_diagonal,
                off_diagonal, lambdas(1);

        // return true if matA is diagonal
        return alpha == 0 && I3 >= 0;
    }
};




#endif //ISO_TLC_SEA_TLC_RESIDUAL_2D_FORMULATION_H
