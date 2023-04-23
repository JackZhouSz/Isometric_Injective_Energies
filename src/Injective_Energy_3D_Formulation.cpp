//
// Created by Charles Du on 11/14/22.
//

#include "Injective_Energy_3D_Formulation.h"
#include <utility>
#include "geo_util.h"

using namespace Eigen;

Injective_Energy_3D_Formulation::Injective_Energy_3D_Formulation(Eigen::MatrixXd restV_, Eigen::Matrix3Xd initV,
                                                                 Eigen::Matrix4Xi T_, const Eigen::VectorXi &handles) :
        Energy_Formulation((initV.cols()-handles.size()) * initV.rows()), T(std::move(T_)),
        V(std::move(initV)), restV(std::move(restV_)) {
    // compute freeI: indices of free vertices
    auto nV = restV.cols();
    int vDim = 3;

    std::vector<bool> freeQ(nV, true);
    for (auto i = 0; i < handles.size(); ++i) {
        freeQ[handles(i)] = false;
    }
    freeI.resize(nV - handles.size());
    {
        int ii = 0;
        for (int i = 0; i < nV; ++i) {
            if (freeQ[i]) {
                freeI[ii] = i;
                ++ii;
            }
        }
    }
    std::sort(freeI.data(), freeI.data() + freeI.size());

    // compute free tet Indices
    // free tet: a tetrahedron with at least one free vertices
    auto nT = T.cols();
    int n_free_tet = 0;
    std::vector<bool> free_tetQ(nT, false);
    for (int i = 0; i < nT; ++i) {
        int i1 = T(0,i);
        int i2 = T(1,i);
        int i3 = T(2,i);
        int i4 = T(3,i);
        if (freeQ[i1] || freeQ[i2] || freeQ[i3] || freeQ[i4]) {
            free_tetQ[i] = true;
            ++n_free_tet;
        }
    }
    free_tetI.resize(n_free_tet);
    free_tet_volumes.resize(n_free_tet);
    {
        int ii = 0;
        for (int i = 0; i < nT; ++i) {
            if (free_tetQ[i]) {
                free_tetI[ii] = i;
                free_tet_volumes[ii] = compute_tet_signed_volume(V.col(T(0,i)),
                                                              V.col(T(1,i)),
                                                              V.col(T(2,i)),
                                                              V.col(T(3,i)));
                ++ii;
            }
        }
    }

    // compute indexDict and T_free
    indexDict = VectorXi::Constant(nV, -1);
    for (auto i = 0; i < freeI.size(); ++i) {
        indexDict(freeI(i)) = i;
    }
    T_free.resize(T.rows(), T.cols());
    for (auto i = 0; i < T.cols(); ++i) {
        for (auto j = 0; j < T.rows(); ++j) {
            T_free(j, i) = indexDict(T(j, i));
        }
    }

    // compute x from V
    curr_x.resize(vDim * freeI.size());
    for (auto i = 0; i < freeI.size(); ++i) {
        int vi = freeI(i);
        for (int j = 0; j < vDim; ++j) {
            curr_x(i * vDim + j) = V(j, vi);
        }
    }

    // extract boundary triangles
    extract_mesh_boundary_triangles(T, boundary_triangles);
    // check if the boundary is fixed
    fixed_boundary = true;
    for (const auto & tri: boundary_triangles) {
        if (freeQ[tri[0]] || freeQ[tri[1]] || freeQ[tri[2]]) {
            fixed_boundary = false;
            break;
        }
    }
    // find boundary vertices
    std::vector<bool> boundary_vertexQ(nV, false);
    for (const auto & tri: boundary_triangles) {
        boundary_vertexQ[tri[0]] = true;
        boundary_vertexQ[tri[1]] = true;
        boundary_vertexQ[tri[2]] = true;
    }
    // build map: vert index -> boundary vert index
    int n_boundary_vert = 0;
    std::vector<int> bndry_vId_of_vert(nV, -1);
    for (int i = 0; i < nV; ++i) {
        if (boundary_vertexQ[i]) {
            bndry_vId_of_vert[i] = n_boundary_vert;
            ++n_boundary_vert;
        }
    }
    // collect boundary vertices
    boundaryI.resize(n_boundary_vert);
    {
        int ii = 0;
        for (int i = 0; i < nV; ++i) {
            if (boundary_vertexQ[i]) {
                boundaryI(ii) = i;
                ++ii;
            }
        }
    }
    // re-index boundary triangles
    for (auto & tri: boundary_triangles) {
        tri[0] = bndry_vId_of_vert[tri[0]];
        tri[1] = bndry_vId_of_vert[tri[1]];
        tri[2] = bndry_vId_of_vert[tri[2]];
    }
}

void Injective_Energy_3D_Formulation::update_x_and_V(const Eigen::VectorXd &x) {
    curr_x = x;
    for (auto i = 0; i < freeI.size(); ++i) {
        int vi = freeI(i);
        for (int j = 0; j < 3; ++j) { // vDim = 3
            V(j, vi) = x(i * 3 + j);
        }
    }
}

bool Injective_Energy_3D_Formulation::set_V(const Eigen::Matrix3Xd &newV) {
    if (newV.cols() != V.cols()) {
        return false;
    }
    V = newV;
    // update x
    curr_x.resize(3 * freeI.size()); // vDim = 3
    for (auto i = 0; i < freeI.size(); ++i) {
        int vi = freeI(i);
        for (int j = 0; j < 3; ++j) {
            curr_x(i * 3 + j) = V(j, vi);
        }
    }
    // update other status by recomputing energy
    VectorXd eList;
    compute_energy(curr_x, eList);
    return true;
}

bool Injective_Energy_3D_Formulation::is_inversion_free() {
    for (double volume : free_tet_volumes) {
        if (volume <= 0) {
            return false;
        }
    }
    return true;
}

bool Injective_Energy_3D_Formulation::met_custom_criterion() {
    if (is_injective()) {
        // store current V when the mesh is injective
        latest_injective_V = V;
        return true;
    } else {
        return false;
    }
}