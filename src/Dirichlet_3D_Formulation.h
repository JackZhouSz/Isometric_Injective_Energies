//
// Created by Charles Du on 4/28/23.
//

#ifndef ISO_TLC_SEA_DIRICHLET_3D_FORMULATION_H
#define ISO_TLC_SEA_DIRICHLET_3D_FORMULATION_H

#include "Distortion_Energy_3D_Formulation.h"

class Dirichlet_3D_Formulation : public Distortion_Energy_3D_Formulation {
public:
    Dirichlet_3D_Formulation(const Eigen::Matrix3Xd& restV_, Eigen::Matrix3Xd initV,
                             Eigen::Matrix4Xi T_, const Eigen::VectorXi &handles);;

    ~Dirichlet_3D_Formulation() override = default;

protected:
    double compute_psi(double I1, double I2, double I3) final { return I2; }

    double compute_psi_with_gradient(double I1, double I2, double I3, Eigen::Vector3d &grad_psi) final {
        grad_psi << 0, 1, 0;
        return I2;
    }

    bool compute_analytic_eigen_information(double sigma1, double sigma2, double sigma3,
                                            double I1, double I2, double I3,
                                            Eigen::Vector<double,9> &lambdas, Eigen::Matrix3d &matA) final {
        lambdas << 2, 2, 2, 2, 2, 2, 2, 2, 2;
        return true;
    }
};


#endif //ISO_TLC_SEA_DIRICHLET_3D_FORMULATION_H
