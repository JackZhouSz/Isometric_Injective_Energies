//
// Created by Charles Du on 11/10/22.
//

#ifndef ISO_TLC_SEA_ENERGY_FORMULATION_H
#define ISO_TLC_SEA_ENERGY_FORMULATION_H

#include <Eigen/Core>
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<double> SpMat;

class Energy_Formulation {
public:
    explicit Energy_Formulation(size_t dim) : input_dimension(dim), curr_x(dim) {};
    virtual ~Energy_Formulation() = default;

    virtual double compute_energy(const Eigen::VectorXd& x, Eigen::VectorXd& energy_list) = 0;

    virtual double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) = 0;

    virtual double compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list,
                                                        Eigen::VectorXd &grad, SpMat &Hess) = 0;

    virtual bool is_injective() { return false; }

    size_t get_input_dimension() const { return input_dimension; }

    // set current x and update related member variables
    bool set_x(const Eigen::VectorXd& x) {
        if (x.size() != input_dimension) return false;
        curr_x = x;
        Eigen::VectorXd eList;
        compute_energy(curr_x, eList);
        return true;
    }

    Eigen::VectorXd get_x() { return curr_x; }

protected:
    size_t input_dimension;
    Eigen::VectorXd curr_x;
};

#endif //ISO_TLC_SEA_ENERGY_FORMULATION_H
