//
// Created by Charles Du on 4/25/23.
//
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include "IO.h"
#include <ghc/filesystem.hpp>
#include "geo_util.h"
#include "NewtonSolver.h"
#include "QuasiNewtonSolver.h"
#include "Dirichlet_2D_Formulation.h"
#include "ARAP_2D_Formulation.h"

using namespace Eigen;


int main(int argc, char const *argv[]) {
    struct {
        std::string input_file;
        std::string distortion_energy = "ARAP";
        bool fix_boundary = false;
        std::string solver_type = "PN";
    } args;
    CLI::App app{"Mapping triangle mesh by optimizing distortion energy"};
    app.add_option("input_file", args.input_file, "input data file")->required();
    app.add_flag("-B,--fix-boundary", args.fix_boundary, "Fix boundary vertices");
    app.add_option("-E,--distortion-energy", args.distortion_energy, "distortion energy type")->required();
    app.add_option("-S,--solver-type", args.solver_type, "solver type");
    CLI11_PARSE(app, argc, argv);
    // save result mesh to the same directory of the input file
    std::string result_file;
    ghc::filesystem::path input_f(args.input_file);
    ghc::filesystem::path input_path = input_f.parent_path();
    ghc::filesystem::path result_f("result.obj");
    result_f = ghc::filesystem::absolute(input_path / result_f);
    result_file = result_f.string();


    // import input data: rest mesh, init mesh, handles
    MatrixXd restV;
    Matrix2Xd initV;
    Matrix3Xi F;
    VectorXi handles;
    if (!import_data(args.input_file, restV, initV, F, handles)) {
        return -1;
    }
    if (args.fix_boundary) {
        std::vector<size_t> boundary_vertices;
        extract_mesh_boundary_vertices(F, boundary_vertices);
        // convert boundary_vertices to a set
        std::set<size_t> boundary_vertices_set(boundary_vertices.begin(), boundary_vertices.end());
        // convert handles to a set
        std::set<size_t> handles_set(handles.data(), handles.data() + handles.size());
        // compute the union of boundary_vertices_set and handles_set
        std::set<size_t> union_set;
        std::set_union(boundary_vertices_set.begin(), boundary_vertices_set.end(),
                       handles_set.begin(), handles_set.end(),
                       std::inserter(union_set, union_set.begin()));
        handles.resize(union_set.size());
        // copy the union set back to handles
        std::copy(union_set.begin(), union_set.end(), handles.data());
    }

    // solver options
    SolverOptions opts;

    // normalize meshes to have unit area
    double init_total_area = abs(compute_total_signed_mesh_area(initV, F));
    initV *= sqrt(1. / init_total_area);
    double rest_total_area = compute_total_unsigned_area(restV, F);
    restV *= sqrt(1. / rest_total_area);


    // initialize energy
    Distortion_Energy_2D_Formulation *energy_ptr;
    if (args.distortion_energy == "ARAP") {
        energy_ptr = new ARAP_2D_Formulation(restV, initV, F, handles);
    } else if (args.distortion_energy == "Dirichlet") {
        energy_ptr = new Dirichlet_2D_Formulation(restV, initV, F, handles);
    } else {
        std::cout << "Unknown distortion energy type: " << args.distortion_energy << std::endl;
        return -1;
    }

    VectorXd x0 = energy_ptr->get_x();

    // minimize energy
    QuasiNewtonSolver QN_Solver;
    NewtonSolver PN_Solver;
    QN_Solver.maxIter = opts.maxIter;
    PN_Solver.maxIter = opts.maxIter;
    if (args.solver_type == "QN") {
        // Quasi-Newton
        QN_Solver.optimize(energy_ptr, x0);
        std::cout << "Quasi-Newton (" << get_stop_type_string(QN_Solver.get_stop_type()) << "), ";
        std::cout << QN_Solver.get_num_iter() << " iterations, " << "E = " << QN_Solver.get_energy() << std::endl;
    } else if (args.solver_type == "PN") {
        // Projected-Newton
        PN_Solver.optimize(energy_ptr, x0);
        std::cout << "Projected-Newton (" << get_stop_type_string(PN_Solver.get_stop_type()) << "), ";
        std::cout << PN_Solver.get_num_iter() << " iterations, " << "E = " << PN_Solver.get_energy() << std::endl;
    } else {
        std::cout << "Unknown solver type: " << args.solver_type << std::endl;
        return -1;
    }

    // save result
    export_mesh(result_file, energy_ptr->get_V(), F);

    return 0;
}