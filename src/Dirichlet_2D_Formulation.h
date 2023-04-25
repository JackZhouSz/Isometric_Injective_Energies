//
// Created by Charles Du on 4/25/23.
//

#ifndef ISO_TLC_SEA_DIRICHLET_2D_FORMULATION_H
#define ISO_TLC_SEA_DIRICHLET_2D_FORMULATION_H

//#include <utility>

#include "Distortion_Energy_2D_Formulation.h"

class Dirichlet_2D_Formulation : public Distortion_Energy_2D_Formulation {
public:
    Dirichlet_2D_Formulation(Eigen::MatrixXd restV_, Eigen::Matrix2Xd initV,
                             Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles)
                             : Distortion_Energy_2D_Formulation(std::move(restV_), std::move(initV), std::move(F_), handles) {};

    ~Dirichlet_2D_Formulation() override = default;

protected:
    double compute_psi(double I1, double I2, double I3) final { return I2; }

    double compute_psi_with_gradient(double I1, double I2, double I3, Eigen::Vector3d &grad_psi) final {
        grad_psi << 0, 1, 0;
        return I2;
    }

    bool compute_analytic_eigen_information(double sigma1, double sigma2,
                                            double I1, double I2, double I3,
                                            Eigen::Vector4d &lambdas, Eigen::Matrix2d &matA) final {
        lambdas << 2, 2, 2, 2;
        return true;
    }
};


#endif //ISO_TLC_SEA_DIRICHLET_2D_FORMULATION_H
