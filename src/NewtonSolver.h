//
// Created by Charles Du on 11/10/22.
//

#ifndef ISO_TLC_SEA_NEWTONSOLVER_H
#define ISO_TLC_SEA_NEWTONSOLVER_H

#include "Solver.h"

class NewtonSolver : public Solver {
public:
    NewtonSolver() = default;
    ~NewtonSolver() override = default;

    void optimize(Energy_Formulation* f, const Eigen::VectorXd& x0) override;

    // backtracking line search
    // Armijo condition: f(x + step_size * p) <= f(x) + gamma * step_size * dot(grad, p)
    // p: line search direction
    // energyList: decomposition of f(x). It helps to robustly compute the difference of f(x) and f(x_next).
    void lineSearch(Energy_Formulation* f, const Eigen::VectorXd &x,
                    const Eigen::VectorXd &p, double &step_size, const Eigen::VectorXd &grad,
                    const Eigen::VectorXd &energyList,
                    double &energy_next, Eigen::VectorXd& x_next) const;

private:
    // linear search parameters
    double gamma = 1e-4;
    double shrink_factor = 0.7;
};

#endif //ISO_TLC_SEA_NEWTONSOLVER_H
