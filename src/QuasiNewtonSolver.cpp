//
// Created by Charles Du on 11/10/22.
//

#include "QuasiNewtonSolver.h"
#include <nlopt.hpp>
#include <iostream>

using namespace Eigen;

double objective_func(const std::vector<double>& x, std::vector<double>& grad, void* solver_func_data)
{
    auto* data = (std::pair<QuasiNewtonSolver*, Energy_Formulation*>*)solver_func_data;
    auto* solver = data->first;
    auto* f = data->second;

    if (solver->stop_type == Injectivity || solver->stop_type == Max_Iter_Reached) {
        if (!grad.empty()) {
            for (double & i : grad) {
                i = 0;
            }
        }
        return solver->curr_energy;
    }

    // compute energy/gradient
    double energy;
    VectorXd g_vec;

    if (grad.empty()) {  // only energy is required
        energy = -1;
        solver->curr_iter++;
        // check injectivity
        if (f->is_injective() && solver->stop_at_injectivity)
        {
            solver->stop_type = Injectivity;
        }
        // max iter criterion
        if (solver->curr_iter >= solver->maxIter) {
            solver->stop_type = Max_Iter_Reached;
        }
    }
    else { // gradient is required
        // convert x to Eigen::VectorXd
        VectorXd x_vec(x.size());
        for (int i = 0; i < x.size(); ++i) {
            x_vec(i) = x[i];
        }
        energy = f->compute_energy_with_gradient(x_vec, g_vec);
        for (int i = 0; i < g_vec.size(); ++i) {
            grad[i] = g_vec(i);
        }
//        solver->lastFunctionValue = energy;
        solver->curr_energy = energy;
    }
    return energy;
}

void QuasiNewtonSolver::optimize(Energy_Formulation *f, const VectorXd &x0) {
    reset();
    if (x0.size() != f->get_input_dimension()) {
        stop_type = Failure;
        return;
    }
    //set algorithm
    nlopt::algorithm algo = nlopt::LD_LBFGS;
    nlopt::opt opt(algo, f->get_input_dimension());

    //set stop criteria
    opt.set_ftol_abs(ftol_abs);
    opt.set_ftol_rel(ftol_rel);
    opt.set_xtol_abs(xtol_abs);
    opt.set_xtol_rel(xtol_rel);
    // bug of NLopt: setting maxeval to -1 other than a positive integer will yield different optimization process
    // instead, we set maxeval to a number much larger than the user-specified max number of iterations
    opt.set_maxeval(maxIter * 100);

    // set energy
    std::pair<QuasiNewtonSolver*, Energy_Formulation*> data_pair(this, f);
    opt.set_min_objective(objective_func, &data_pair);
    // set x
    curr_x = x0;
    std::vector<double> x(x0.size());
    for (int i = 0; i < x.size(); ++i) {
        x[i] = x0(i);
    }
    double minf;

    //optimize
    try {
        curr_iter = 0;
        nlopt::result result = opt.optimize(x, minf);
        // save x to curr_x
        for (size_t i=0; i < x.size(); ++i) {
            curr_x(i) = x[i];
        }
        // store result x in the energy object f
        f->set_x(curr_x);
        assert(curr_energy == minf);
        switch (result) {
            case nlopt::SUCCESS:
                if (stop_type!=Injectivity && stop_type != Max_Iter_Reached) {
                    stop_type = StopType::Success;
                }
                break;
            case nlopt::STOPVAL_REACHED:
                break;
            case nlopt::FTOL_REACHED:
                stop_type = Ftol_Reached;
                break;
            case nlopt::XTOL_REACHED:
                stop_type = Xtol_Reached;
                break;
            case nlopt::MAXEVAL_REACHED:
                break;
            case nlopt::MAXTIME_REACHED:
                break;
            default:
//                std::cout << "unexpected return code!" << std::endl;
                break;
        }
    }
    catch (std::exception& e) {
        std::cout << "nlopt failed: " << e.what() << std::endl;
        stop_type = Failure;
    }
}
