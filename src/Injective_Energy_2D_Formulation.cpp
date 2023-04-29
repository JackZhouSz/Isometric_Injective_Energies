//
// Created by Charles Du on 11/12/22.
//

#include "Injective_Energy_2D_Formulation.h"

#include <utility>
#include "geo_util.h"

using namespace Eigen;

Injective_Energy_2D_Formulation::Injective_Energy_2D_Formulation(Eigen::MatrixXd restV_, Eigen::Matrix2Xd initV,
                                                                 Eigen::Matrix3Xi F_, const Eigen::VectorXi &handles)
        : Energy_Formulation((initV.cols() - handles.size()) * initV.rows()),
          F(std::move(F_)), V(std::move(initV)), restV(std::move(restV_)) {
    // compute freeI: indices of free vertices
    auto nV = restV.cols();
    const int vDim = 2;
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

    // free face: a triangle with at least one free vertex
    auto nF = F.cols();
    int n_free_face = 0;
    std::vector<bool> free_faceQ(nF, false);
    for (int i = 0; i < nF; ++i) {
        int i1 = F(0, i);
        int i2 = F(1, i);
        int i3 = F(2, i);
        if (freeQ[i1] || freeQ[i2] || freeQ[i3]) {
            free_faceQ[i] = true;
            ++n_free_face;
        }
    }
    free_faceI.resize(n_free_face);
    free_face_areas.resize(n_free_face);
    {
        int ii = 0;
        for (int i = 0; i < nF; ++i) {
            if (free_faceQ[i]) {
                free_faceI[ii] = i;
                free_face_areas[ii] = compute_tri_signed_area(V.col(F(0,i)),
                                                              V.col(F(1,i)),
                                                              V.col(F(2,i)));
                ++ii;
            }
        }
    }

    // compute indexDict and F_free
    indexDict = VectorXi::Constant(nV, -1);
    for (auto i = 0; i < freeI.size(); ++i) {
        indexDict(freeI(i)) = i;
    }
    F_free.resize(F.rows(), F.cols());
    for (auto i = 0; i < F.cols(); ++i) {
        for (auto j = 0; j < F.rows(); ++j) {
            F_free(j, i) = indexDict(F(j, i));
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

    // extract boundary edges
    extract_mesh_boundary_edges(F, boundary_edges);
    // check if the boundary is fixed
    fixed_boundary = true;
    for (const auto &edge: boundary_edges) {
        if (freeQ[edge.first] || freeQ[edge.second]) {
            fixed_boundary = false;
            break;
        }
    }
    // find boundary vertices
    std::vector<bool> is_boundary_vert(V.cols(), false);
    for (auto &e: boundary_edges) {
        is_boundary_vert[e.first] = true;
        is_boundary_vert[e.second] = true;
    }
    // build map: vert index -> boundary vert index
    int n_boundary_vert = 0;
    std::vector<int> bndry_vId_of_vert(V.cols(), -1);
    for (int i = 0; i < is_boundary_vert.size(); ++i) {
        if (is_boundary_vert[i]) {
            bndry_vId_of_vert[i] = n_boundary_vert;
            ++n_boundary_vert;
        }
    }
    // collect boundary vertices
    boundaryI.resize(n_boundary_vert);
    {
        int ii = 0;
        for (int i = 0; i < bndry_vId_of_vert.size(); ++i) {
            if (bndry_vId_of_vert[i] != -1) {
                boundaryI(ii) = i;
                ++ii;
            }
        }
    }
    // re-index boundary edges into boundary vertices
    for (auto& e : boundary_edges) {
        e.first = bndry_vId_of_vert[e.first];
        e.second = bndry_vId_of_vert[e.second];
    }
}


void Injective_Energy_2D_Formulation::update_x_and_V(const VectorXd &x) {
    curr_x = x;
    int vDim = 2;
    for (auto i = 0; i < freeI.size(); ++i) {
        for (auto j = 0; j < vDim; ++j) {
            V(j, freeI(i)) = x[i * vDim + j];
        }
    }
}

bool Injective_Energy_2D_Formulation::set_V(const Matrix2Xd &vertices) {
    if (vertices.cols() != V.cols()) return false;
    V = vertices;
    int vDim = 2;
    // update current x
    curr_x.resize(freeI.size() * vDim);
    for (auto i = 0; i < freeI.size(); ++i) {
        for (auto j = 0; j < vDim; ++j) {
            curr_x[i * vDim + j] = V(j, freeI(i));
        }
    }
    // update other status by recomputing energy
    VectorXd eList;
    compute_energy(curr_x, eList);
    return true;
}

bool Injective_Energy_2D_Formulation::is_inversion_free() {
    for (double area: free_face_areas) {
        if (area <= 0) return false;
    }
    return true;
}

bool Injective_Energy_2D_Formulation::met_custom_criterion() {
    if (is_injective()) {
        // store current V when the mesh is injective
        latest_injective_V = V;
        return true;
    } else {
        return false;
    }
}
