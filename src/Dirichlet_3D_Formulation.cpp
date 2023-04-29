//
// Created by Charles Du on 4/28/23.
//

#include "Dirichlet_3D_Formulation.h"

Dirichlet_3D_Formulation::Dirichlet_3D_Formulation(const Eigen::Matrix3Xd& restV_, Eigen::Matrix3Xd initV, Eigen::Matrix4Xi T_,
                                                   const Eigen::VectorXi &handles)
        : Distortion_Energy_3D_Formulation(restV_, std::move(initV), std::move(T_), handles) {}
