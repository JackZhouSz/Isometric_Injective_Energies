//
// Created by Charles Du on 11/10/22.
//

#ifndef ISO_TLC_SEA_QUASINEWTONSOLVER_H
#define ISO_TLC_SEA_QUASINEWTONSOLVER_H

#include "Solver.h"

class QuasiNewtonSolver : public Solver {
public:
    QuasiNewtonSolver() = default;
    ~QuasiNewtonSolver() override = default;

    void optimize(Energy_Formulation* f, const Eigen::VectorXd& x0) override;

private:
    // a helper function needed by NLopt solver
    friend double objective_func(const std::vector<double>& x, std::vector<double>& grad, void* solver_func_data);
};

#endif //ISO_TLC_SEA_QUASINEWTONSOLVER_H
