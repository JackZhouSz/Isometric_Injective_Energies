//
// Created by Charles Du on 11/9/22.
//
#include <CLI/CLI.hpp>
#include <Eigen/Core>
#include "IO.h"
#include <ghc/filesystem.hpp>
#include "geo_util.h"
#include "NewtonSolver.h"
#include "QuasiNewtonSolver.h"
#include "TLC_2D_Formulation.h"

using namespace Eigen;


int main(int argc, char const* argv[]) {
    struct
    {
        std::string input_file;
        bool stop_at_injectivity = false;
        std::string solver_options_file;
        std::string result_file;
    } args;
    CLI::App app{"Mapping triangle mesh by optimizing TLC energy"};
    app.add_option("input_file", args.input_file, "input data file")->required();
    app.add_flag("-I,--stop-at-injectivity", args.stop_at_injectivity, "Stop optimization when injective map is found");
    app.add_option("solver_options_file", args.solver_options_file, "solver options file");
    app.add_option("result_file", args.result_file, "result file");
    CLI11_PARSE(app, argc, argv);
    if (args.result_file.empty()) {
        // if result_file is not given, save result to input_path/result
        namespace fs = ghc::filesystem;
        fs::path input_f(args.input_file);
        fs::path input_path = input_f.parent_path();
        fs::path result_f("result.obj");
        result_f = fs::absolute(input_path / result_f);
        args.result_file = result_f.string();
    }

    // import input data: rest mesh, init mesh, handles
    MatrixXd restV;
    Matrix2Xd initV;
    Matrix3Xi F;
    VectorXi handles;
    if(!import_data(args.input_file, restV, initV, F, handles)) {
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
//    restV *= scale;
    double rest_total_area = compute_total_unsigned_area(restV, F);
    restV *= sqrt(1. / rest_total_area);


    // initialize energy
    TLC_2D_Formulation energy(restV, initV, F, handles, opts.form, opts.alpha);
    VectorXd x0 = energy.get_x();

    // minimize energy
    QuasiNewtonSolver QN_Solver;
    NewtonSolver PN_Solver;
    QN_Solver.maxIter = opts.maxIter;
    PN_Solver.maxIter = opts.maxIter;
    // stage 1: find injectivity
    bool injective_found = true;
    QN_Solver.stop_at_injectivity = true;
    QN_Solver.optimize(&energy, x0);
    if (!energy.is_injective()) {
        PN_Solver.stop_at_injectivity = true;
        PN_Solver.optimize(&energy, x0);
        if (!energy.is_injective()) {
            std::cout << "Failed to find injective map!" << std::endl;
            injective_found = false;
        }
    }
    // stage 2: lower distortion
    if (injective_found && !args.stop_at_injectivity) {
        // optimize until energy convergence, starting from the result of stage 1
        PN_Solver.stop_at_injectivity = false;
        PN_Solver.optimize(&energy, energy.get_x());
    }

    // save result
    //export_result(args.result_file, energy.get_latest_injective_V());
//    export_result(args.result_file,
//                  TLC_energy.get_latest_injective_V() * sqrt(init_total_area));
    if (injective_found) {
        export_mesh(args.result_file, energy.get_latest_injective_V(), F);
    } else {
        export_mesh(args.result_file, energy.get_V(), F);
    }

    return 0;
}
