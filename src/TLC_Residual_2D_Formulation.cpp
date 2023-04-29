//
// Created by Charles Du on 4/29/23.
//

#include "TLC_Residual_2D_Formulation.h"
#include <iostream>
#include "deformation_gradient_util.h"
#include <Eigen/Eigenvalues>

TLC_Residual_2D_Formulation::TLC_Residual_2D_Formulation(const Eigen::MatrixXd &restV, Eigen::Matrix2Xd initV,
                                                         Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles,
                                                         std::string form_, double alpha_)
                                                         : Distortion_Energy_2D_Formulation(restV, std::move(initV), std::move(F_), handles),
                                                           form(std::move(form_)), alpha(alpha_) {
    assert(form == "Tutte" || form == "harmonic");
    assert(alpha >= 0);

    // print formulation parameters
    std::cout << "form: " << form << std::endl;
    std::cout << "alpha: " << alpha << std::endl;

    // compute restA, rest_invEdgeMat, pFpx_list if form == "Tutte"
    double total_rest_area = restA.sum();
    if (form == "Tutte") {
        // we assume the auxiliary triangles are equilateral triangles
        // whose total measure is the same as the input rest mesh.
        double regular_tri_area = total_rest_area / F.cols();
        double regular_tri_squared_length = 4 * regular_tri_area / sqrt(3.0);
        restA = Eigen::VectorXd::Constant(F.cols(), regular_tri_area);
        // compute inverse of edge matrix and pFpx
        Eigen::Vector2d v1(0, 0);
        double regular_tri_edge_length = sqrt(regular_tri_squared_length);
        Eigen::Vector2d v2(regular_tri_edge_length, 0);
        Eigen::Vector2d v3(regular_tri_edge_length / 2, sqrt(3.) * regular_tri_edge_length / 2);
        Eigen::Matrix2d edgeMat;
        compute_edge_matrix(v1, v2, v3, edgeMat);
        Eigen::Matrix2d invEdgeMat = edgeMat.inverse();
        Eigen::MatrixXd pFpx;
        compute_flatten_pFpx(invEdgeMat, pFpx);
        rest_invEdgeMat.resize(F.cols(), invEdgeMat);
        pFpx_list.resize(F.cols(), pFpx);
    }
    // if form == "harmonic",
    // the restA, rest_invEdgeMat, pFpx_list are already computed in the base class.
}