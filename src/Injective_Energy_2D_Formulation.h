//
// Created by Charles Du on 11/12/22.
//

#ifndef ISO_TLC_SEA_INJECTIVE_ENERGY_2D_FORMULATION_H
#define ISO_TLC_SEA_INJECTIVE_ENERGY_2D_FORMULATION_H

#include "Energy_Formulation.h"

class Injective_Energy_2D_Formulation : public Energy_Formulation {
public:
    Injective_Energy_2D_Formulation(Eigen::MatrixXd restV_, Eigen::Matrix2Xd initV,
                                    Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles);

    ~Injective_Energy_2D_Formulation() override = default;

    double compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) override = 0;

    double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override = 0;

    double compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list,
                                                Eigen::VectorXd &grad, SpMat &Hess) override = 0;

    // check whether the current mesh is injective.
    // If so, store the current mesh.
    bool is_injective() override = 0;

    // get the mesh vertices when the last time the mesh is injective
    Eigen::Matrix2Xd get_latest_injective_V() { return latest_injective_V; }

    // get the current mesh vertices
    Eigen::Matrix2Xd get_V() { return V; }

    // set current mesh vertices
    bool set_V(const Eigen::Matrix2Xd& vertices);

protected:
    // rest vertices
    Eigen::MatrixXd restV;
    // current V of target mesh
    Eigen::Matrix2Xd V;
    // V indices of triangles
    Eigen::Matrix3Xi F;
    // indices of free vertices
    Eigen::VectorXi freeI;
    // indices of free faces (i.e. face with at least one free vertex)
    Eigen::VectorXi free_faceI;
    // signed area of free faces
    Eigen::VectorXd free_face_areas;
    // map: V index --> freeV index. If V(i) is not free, indexDict(i) = -1.
    Eigen::VectorXi indexDict;
    // freeV indices of triangles.
    Eigen::Matrix3Xi F_free;

    // boundary vertex indices
    Eigen::VectorXi boundaryI;
    // boundary edges: each edge is a pair of indices of boundaryI
    std::vector<std::pair<size_t, size_t>> boundary_edges;
    // whether mesh boundary is fixed
    bool fixed_boundary;

    // V when the last time is_injective() is true
    Eigen::Matrix2Xd latest_injective_V;

    // update current x and V
    void update_x_and_V(const Eigen::VectorXd &x);
};


#endif //ISO_TLC_SEA_INJECTIVE_ENERGY_2D_FORMULATION_H
