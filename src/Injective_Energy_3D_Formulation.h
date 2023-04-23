//
// Created by Charles Du on 11/14/22.
//

#ifndef ISO_TLC_SEA_INJECTIVE_ENERGY_3D_FORMULATION_H
#define ISO_TLC_SEA_INJECTIVE_ENERGY_3D_FORMULATION_H

#include "Energy_Formulation.h"

class Injective_Energy_3D_Formulation : public Energy_Formulation {
public:
    Injective_Energy_3D_Formulation(Eigen::MatrixXd restV_, Eigen::Matrix3Xd initV,
            Eigen::Matrix4Xi T_, const Eigen::VectorXi &handles);

    ~Injective_Energy_3D_Formulation() override = default;

    double compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) override = 0;

    double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override = 0;

    double compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list,
                                                Eigen::VectorXd &grad, SpMat &Hess) override = 0;

    // check whether the current mesh is injective.
    // If so, store the current mesh.
    bool met_custom_criterion() override;

    // check whether the current mesh is free of inverted or degenerated tetrahedrons
    bool is_inversion_free();

    // check whether the current mesh is injective
    // by default, it checks whether the current mesh is inversion free,
    // derived class can override this function to implement other injectivity check
    virtual bool is_injective() { return is_inversion_free(); };

    // get the mesh vertices when the last time the mesh is injective
    Eigen::Matrix3Xd get_latest_injective_V() { return latest_injective_V; }

    // get the current mesh vertices
    Eigen::Matrix3Xd get_V() { return V; }

    // set current mesh vertices
    bool set_V(const Eigen::Matrix3Xd& vertices);

protected:
    // rest vertices
    Eigen::MatrixXd restV;
    // current V of target mesh
    Eigen::Matrix3Xd V;
    // V indices of tetrahedrons
    Eigen::Matrix4Xi T;
    // indices of free vertices
    Eigen::VectorXi freeI;
    // indices of free tetrahedrons (i.e. tetrahedron with at least one free vertex)
    Eigen::VectorXi free_tetI;
    // signed volume of free tetrahedrons
    Eigen::VectorXd free_tet_volumes;
    // map: V index --> freeV index. If V(i) is not free, indexDict(i) = -1.
    Eigen::VectorXi indexDict;
    // freeV indices of tetrahedrons.
    Eigen::Matrix4Xi T_free;

    // boundary vertex indices
    Eigen::VectorXi boundaryI;
    // boundary edges: each edge is a pair of indices of boundaryI
//    std::vector<std::pair<size_t, size_t>> boundary_edges;
    // boundary triangles: each triangle is a triplet of indices of boundaryI
    std::vector<std::array<int,3>> boundary_triangles;
    // whether mesh boundary is fixed
    bool fixed_boundary;

    // V when the last time met_custom_criterion() is true
    Eigen::Matrix3Xd latest_injective_V;

    // update current x and V
    void update_x_and_V(const Eigen::VectorXd &x);

};


#endif //ISO_TLC_SEA_INJECTIVE_ENERGY_3D_FORMULATION_H
