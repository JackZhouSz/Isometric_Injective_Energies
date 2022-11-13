//
// Created by Charles Du on 11/10/22.
//

#include "NewtonSolver.h"
#include <Eigen/CholmodSupport>
#include <iostream>

using namespace Eigen;
typedef Eigen::CholmodSupernodalLLT<SpMat> CholmodSolver;

void
NewtonSolver::lineSearch(Energy_Formulation *f, const VectorXd &x, const VectorXd &p, double &step_size,
                         const VectorXd &grad, const VectorXd &energyList,
                         double &energy_next, VectorXd& x_next) const {
    double gp = gamma * grad.transpose() * p;
    x_next = x + step_size * p;
    VectorXd energyList_next;
    energy_next = f->compute_energy(x_next, energyList_next);

    VectorXd energy_diff_list = energyList_next - energyList;
    double energy_diff = energy_diff_list.sum();

    while (energy_diff > step_size * gp) {
        step_size *= shrink_factor;
        x_next = x + step_size * p;
        energy_next = f->compute_energy(x_next, energyList_next);

        energy_diff_list = energyList_next - energyList;
        energy_diff = energy_diff_list.sum();
    }
}

void NewtonSolver::optimize(Energy_Formulation *f, const VectorXd &x0) {
    reset();
    if (x0.size() != f->get_input_dimension()) {
        stop_type = Failure;
        return;
    }
    VectorXd& x = curr_x;
    x = x0;
    VectorXd x_next = x;
    double& energy = curr_energy;
    double energy_next;
    VectorXd energyList;
    VectorXd grad(x.size());
    SpMat mat(x.size(), x.size());

    //first iter: initialize solver
    curr_iter = 0;
    energy = f->compute_energy_with_gradient_Hessian(x,energyList,grad,mat);
    // check termination before linear search
    if (f->is_injective() && stop_at_injectivity)
    {
        // no matter whether stop_at_injectivity, f->is_injective() is always called
        // this helps the object f to update the most recent injective mesh
        stop_type = Injectivity;
        return;
    }
    if (grad.norm() < gtol) {
        stop_type = Gtol_Reached;
        return;
    }
    // initialize linear solver
    CholmodSolver solver;
    solver.analyzePattern(mat);
    // compute Newton search direction p
    solver.factorize(mat);
    if (solver.info() != ComputationInfo::Success) {
        std::cout << "iter 0: decomposition failed" << std::endl;
        stop_type = Failure;
        return;
    }
    VectorXd p = solver.solve(-grad);
    if (solver.info() != ComputationInfo::Success) {
        std::cout << "iter 0: solving failed" << std::endl;
        stop_type = Failure;
        return;
    }
    // backtracking line search
    double step_size = 1.0;
    double xnorm = x.norm();
    lineSearch(f,x, p, step_size, grad, energyList, energy_next, x_next);
    // check termination after line search
    double step_norm = p.norm() * step_size;
    if (is_stagnant(energy, energy_next, xnorm, step_norm, stop_type)) {
        f->set_x(x); // make sure the energy object f stores the result x
        return;
    }

    // iteration 1, 2, ...
    for (curr_iter = 1; curr_iter < maxIter; ++curr_iter) {
        x = x_next;
        energy = f->compute_energy_with_gradient_Hessian(x,energyList,grad,mat);
        // check before line search
        if (f->is_injective() && stop_at_injectivity) {
            // no matter whether stop_at_injectivity, f->is_injective() is always called
            // this helps the object f to update the most recent injective mesh
            stop_type = Injectivity;
            return;
        }
        if (grad.norm() < gtol) {
            stop_type = Gtol_Reached;
            return;
        }
        // compute Newton search direction p
        solver.factorize(mat);
        if (solver.info() != ComputationInfo::Success) {
            std::cout << "iter " << curr_iter << ": decomposition failed" << std::endl;
            stop_type = Failure;
            return;
        }
        p = solver.solve(-grad);
        if (solver.info() != ComputationInfo::Success) {
            std::cout << "iter " << curr_iter << ": solving failed" << std::endl;
            stop_type = Failure;
            return;
        }
        // backtracking line search
        step_size = 1.0;
        xnorm = x.norm();
        lineSearch(f,x, p, step_size, grad, energyList, energy_next, x_next);
        // check termination after line search
        step_norm = p.norm() * step_size;
        if (is_stagnant(energy, energy_next, xnorm, step_norm, stop_type)) {
            f->set_x(x);
            return;
        }
    }

    // reach max iterations
    stop_type = Max_Iter_Reached;
}