//
// Created by Charles Du on 4/28/23.
//

#include "ARAP_3D_Formulation.h"

#include <utility>

ARAP_3D_Formulation::ARAP_3D_Formulation(const Eigen::Matrix3Xd &restV_, Eigen::Matrix3Xd initV, Eigen::Matrix4Xi T_,
                                         const Eigen::VectorXi &handles) : Distortion_Energy_3D_Formulation(restV_,
                                                                                                            std::move(initV), std::move(T_),
                                                                                                            handles) {

}
