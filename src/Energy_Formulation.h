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
    explicit Energy_Formulation(size_t dim) : input_dimension(dim) {};
    virtual ~Energy_Formulation() = default;

    virtual double compute_energy(const Eigen::VectorXd& x, Eigen::VectorXd& energy_list) = 0;

    virtual double compute_energy_with_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &grad) = 0;

    virtual double compute_energy_with_gradient_Hessian(const Eigen::VectorXd &x, Eigen::VectorXd &energy_list,
                                                        Eigen::VectorXd &grad, SpMat &Hess) = 0;

    virtual bool is_injective() { return false; }

    size_t get_input_dimension() { return input_dimension; }

private:
    size_t input_dimension;
};

#endif //ISO_TLC_SEA_ENERGY_FORMULATION_H
