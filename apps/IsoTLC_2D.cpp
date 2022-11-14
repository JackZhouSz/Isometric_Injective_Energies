//
// Created by Charles Du on 11/13/22.
//
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include "IO.h"
#include <ghc/filesystem.hpp>
#include "geo_util.h"
#include "NewtonSolver.h"
#include "QuasiNewtonSolver.h"
#include "IsoTLC_2D_Formulation.h"

using namespace Eigen;


int main(int argc, char const *argv[]) {
    struct {
        std::string input_file;
        bool stop_at_injectivity = false;
        std::string solver_options_file;
    } args;
    CLI::App app{"Mapping triangle mesh by optimizing IsoTLC energy"};
    app.add_option("input_file", args.input_file, "input data file")->required();
    app.add_flag("-I,--stop-at-injectivity", args.stop_at_injectivity, "Stop optimization when injective map is found");
    app.add_option("solver_options_file", args.solver_options_file, "solver options file");
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

    // import options
    SolverOptions opts;
    if (!args.solver_options_file.empty()) {
        import_solver_options(args.solver_options_file, opts);
    }

    // normalize meshes to have unit area
    double init_total_area = abs(compute_total_signed_mesh_area(initV, F));
    initV *= sqrt(1. / init_total_area);
    double rest_total_area = compute_total_unsigned_area(restV, F);
    restV *= sqrt(1. / rest_total_area);


    // initialize energy
    IsoTLC_2D_Formulation energy(restV, initV, F, handles, opts.form, opts.alpha);
    VectorXd x0 = energy.get_x();

    // minimize energy
    QuasiNewtonSolver QN_Solver;
    NewtonSolver PN_Solver;
    QN_Solver.maxIter = opts.maxIter;
    PN_Solver.maxIter = opts.maxIter;
    // stage 1: find injectivity
    std::cout << "------ stage 1: find injectivity ------" << std::endl;
    bool injective_found = true;
    QN_Solver.stop_at_injectivity = true;
    QN_Solver.optimize(&energy, x0);
    std::cout << "Quasi-Newton (" << get_stop_type_string(QN_Solver.get_stop_type()) << "), ";
    std::cout << QN_Solver.get_num_iter() << " iterations, " << "E = " << QN_Solver.get_energy() << std::endl;
    if (!energy.is_injective()) {
        PN_Solver.stop_at_injectivity = true;
        PN_Solver.optimize(&energy, x0);
        std::cout << "Projected-Newton (" << get_stop_type_string(PN_Solver.get_stop_type()) << "), ";
        std::cout << PN_Solver.get_num_iter() << " iterations, " << "E = " << PN_Solver.get_energy() << std::endl;
        if (!energy.is_injective()) {
            std::cout << "Failed to find injective map!" << std::endl;
            injective_found = false;
        }
    }
    // stage 2: lower distortion
    if (injective_found && !args.stop_at_injectivity) {
        std::cout << "------ stage 2: lower distortion ------" << std::endl;
        // optimize until energy convergence, starting from the result of stage 1
        PN_Solver.stop_at_injectivity = false;
        PN_Solver.optimize(&energy, energy.get_x());
        std::cout << "Projected-Newton (" << get_stop_type_string(PN_Solver.get_stop_type()) << "), ";
        std::cout << PN_Solver.get_num_iter() << " iterations, " << "E = " << PN_Solver.get_energy() << std::endl;
    }


    // save result
    if (injective_found) {
        export_mesh(result_file, energy.get_latest_injective_V(), F);
    } else {
        export_mesh(result_file, energy.get_V(), F);
    }

    return 0;
}