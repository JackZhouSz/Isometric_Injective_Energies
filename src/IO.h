//
// Created by Charles Du on 11/9/22.
//

#ifndef ISO_TLC_SEA_IO_H
#define ISO_TLC_SEA_IO_H

#include <Eigen/Core>

struct SolverOptions {
    // Parameters for energy
    // form of auxiliary simplices
    // Tutte: regular simplices
    // harmonic: simplices of rest mesh
    std::string form = "harmonic";
    // lifting parameter
    double alpha = 1e-4;
    // center angle for circular arcs (SEA, IsoSEA)
    double theta = 0.1;

    // Parameters for solver
    // maximum number of iterations
    size_t maxIter = 10000;
};

// import 2D input data
bool import_data(const std::string& filename,
                Eigen::MatrixXd &restV,
                Eigen::Matrix2Xd &initV,
                Eigen::Matrix3Xi &F,
                Eigen::VectorXi &handles
);

// import 3D input data
bool import_data(const std::string& filename,
                  Eigen::Matrix3Xd &restV,
                  Eigen::Matrix3Xd &initV,
                  Eigen::Matrix4Xi &F,
                  Eigen::VectorXi &handles
);

// import solver options
void import_solver_options(const std::string &filename, SolverOptions& options);

// export mesh
void export_mesh(const std::string &filename, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);


#endif //ISO_TLC_SEA_IO_H
