//
// Created by Charles Du on 11/10/22.
//

#include "Energy_Formulation.h"
#include "NewtonSolver.h"
#include "QuasiNewtonSolver.h"
#include <iostream>

class Test_function : public Energy_Formulation {
public:
    explicit Test_function(const Eigen::VectorXd& x0) : Energy_Formulation(x0.size()) {}
    ~Test_function() override = default;

    // f(x) = (x[0]^2 + 2 x[1]^2 + 3 x[2]^2 + ...) /2
    double compute_energy(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list) override {
        energy_list.resize(x.size());
        for (int i = 0; i < x.size(); ++i) {
            energy_list(i) = (i+1) * x(i) * x(i);
        }
        energy_list *= 0.5;
        return energy_list.sum();
    }

    double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) override {
        Eigen::VectorXd energy_list;
        energy_list.resize(x.size());
        grad.resize(x.size());
        for (int i = 0; i < x.size(); ++i) {
            energy_list(i) = (i+1) * x(i) * x(i);
            grad(i) = (i+1) * x(i);
        }
        energy_list *= 0.5;
        return energy_list.sum();
    }

    double compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list,
                                                Eigen::VectorXd &grad, SpMat &Hess) override
    {
        energy_list.resize(x.size());
        grad.resize(x.size());
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(x.size());
        for (int i = 0; i < x.size(); ++i) {
            energy_list(i) = (i+1) * x(i) * x(i);
            grad(i) = (i+1) * x(i);
            tripletList.emplace_back(i, i, i+1);
        }
        energy_list *= 0.5;
        Hess.resize(x.size(), x.size());
        Hess.setFromTriplets(tripletList.begin(), tripletList.end());
        return energy_list.sum();
    }
};

int main() {
    Eigen::VectorXd x;
    x.setRandom(500);
    Test_function f(x);
    Eigen::VectorXd energyList;
    std::cout << "Init, ";
    std::cout << "f = " << f.compute_energy(x, energyList) << ", ";
    std::cout << "||x|| = " << x.norm() << std::endl;

    QuasiNewtonSolver qn_solver;
    qn_solver.optimize(&f, x);
    std::cout << "Quasi-Newton (" << get_stop_type_string(qn_solver.get_stop_type()) << "), ";
    std::cout << qn_solver.get_num_iter() << " iterations, ";
    std::cout << "f = " << qn_solver.get_energy() << ", ";
    std::cout << "||x|| = " << qn_solver.get_x().norm() << std::endl;

    NewtonSolver n_solver;
    n_solver.optimize(&f, x);
    std::cout << "Newton (" << get_stop_type_string(n_solver.get_stop_type()) << "), ";
    std::cout << n_solver.get_num_iter() << " iterations, ";
    std::cout << "f = " << n_solver.get_energy() << ", ";
    std::cout << "||x|| = " << n_solver.get_x().norm() << std::endl;

    return 0;
}