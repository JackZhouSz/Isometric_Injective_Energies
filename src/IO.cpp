//
// Created by Charles Du on 11/9/22.
//

#include "IO.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>


bool import_data(const std::string& filename, Eigen::MatrixXd &restV, Eigen::Matrix2Xd &initV, Eigen::Matrix3Xi &F,
                 Eigen::VectorXi &handles) {
    using json = nlohmann::json;
    std::ifstream fin(filename.c_str());
    if (!fin) {
        std::cout << "Failed to open input data file!" << std::endl;
        return false;
    }
    json data;
    fin >> data;
    fin.close();

    // rest vertices
    assert(data.contains("restV"));
    size_t n = data["restV"].size();
    assert(n>0);
    size_t ndim = data["restV"][0].size();
    restV.resize(ndim, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ndim; ++j) {
            restV(j, i) = data["restV"][i][j].get<double>();
        }
    }

    //initial vertices
    assert(data.contains("initV"));
    n = data["initV"].size();
    assert(n>0);
    ndim = data["initV"][0].size();
    assert(ndim == 2);
    initV.resize(ndim, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ndim; ++j) {
            initV(j, i) = data["initV"][i][j].get<double>();
        }
    }

    // F
    size_t simplexSize;
    assert(data.contains("F"));
    n = data["F"].size();
    assert(n>0);
    simplexSize = data["F"][0].size();
    assert(simplexSize == 3);
    F.resize(simplexSize, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < simplexSize; ++j) {
            F(j, i) = data["F"][i][j].get<int>();
        }
    }

    // handles
    assert(data.contains("handles"));
    n = data["handles"].size();
    handles.resize(n);
    for (int i = 0; i < n; ++i) {
        handles(i) = data["handles"][i].get<int>();
    }

    return true;
}

bool import_data(const std::string &filename, Eigen::Matrix3Xd &restV, Eigen::Matrix3Xd &initV, Eigen::Matrix4Xi &F,
                 Eigen::VectorXi &handles) {
    using json = nlohmann::json;
    std::ifstream fin(filename.c_str());
    if (!fin) {
        std::cout << "Failed to open input data file!" << std::endl;
        return false;
    }
    json data;
    fin >> data;
    fin.close();

    // rest vertices
    assert(data.contains("restV"));
    size_t n = data["restV"].size();
    assert(n>0);
    size_t ndim = data["restV"][0].size();
    restV.resize(ndim, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ndim; ++j) {
            restV(j, i) = data["restV"][i][j].get<double>();
        }
    }

    //initial vertices
    assert(data.contains("initV"));
    n = data["initV"].size();
    assert(n>0);
    ndim = data["initV"][0].size();
    assert(ndim == 3);
    initV.resize(ndim, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ndim; ++j) {
            initV(j, i) = data["initV"][i][j].get<double>();
        }
    }

    // F
    size_t simplexSize;
    assert(data.contains("F"));
    n = data["F"].size();
    assert(n>0);
    simplexSize = data["F"][0].size();
    assert(simplexSize == 4);
    F.resize(simplexSize, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < simplexSize; ++j) {
            F(j, i) = data["F"][i][j].get<int>();
        }
    }

    // handles
    assert(data.contains("handles"));
    n = data["handles"].size();
    handles.resize(n);
    for (int i = 0; i < n; ++i) {
        handles(i) = data["handles"][i].get<int>();
    }

    return true;
}

void import_solver_options(const std::string &filename, SolverOptions& options) {
    using json = nlohmann::json;
    std::ifstream fin(filename.c_str());
    if (!fin) {
        throw std::runtime_error("Solver option file does not exist!");
    }
    json data;
    fin >> data;
    fin.close();

    options.form = data["form"];
    options.alpha = data["alpha"];
    if (data.contains("theta")) {
        options.theta = data["theta"];
    } else {
        options.theta = 0.1;
    }

    options.maxIter = data["maxIter"];
}

void export_mesh(const std::string &filename, const Eigen::MatrixXd &V, const Eigen::MatrixXi &F) {
    std::ofstream fout(filename.c_str());
    if (!fout) {
        throw std::runtime_error("Failed to open mesh file!");
    }
    // vertices
    auto nv = V.cols();
    auto ndim = V.rows();
    for (int i = 0; i < nv; ++i)
    {
        fout << "v ";
        for (int j = 0; j < ndim; ++j)
        {
            fout <<  V(j, i) << " ";
        }
        if (ndim == 2) {
            fout << "0";
        }
        fout << std::endl;
    }
    // simplices
    auto nf = F.cols();
    auto nSimplexSize = F.rows();
    for (int i = 0; i < nf; ++i)
    {
        fout << "f ";
        for (int j = 0; j < nSimplexSize; ++j)
        {
            // obj file: vertex index starts from 1
            fout <<  1 + F(j, i) << " ";
        }
        fout << std::endl;
    }

    fout.close();
}
