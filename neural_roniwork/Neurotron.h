#pragma once

#include <Eigen/Dense>

using namespace Eigen;

class Neurotron {
private:
	const int	CLASS_1 = 0b001,
				CLASS_2 = 0b010,
				CLASS_3 = 0b100;

	MatrixXd	m_net,
				m_weight,
				m_bias,
				m_output;

	MatrixXd	m_target_out, 
				m_target_class;

	MatrixXd m_activate(MatrixXd m);
	MatrixXd m_activate_d(MatrixXd m);
	MatrixXd m_init_weight(double max, int width, int height);
public:
	Neurotron();
	void init(int cols, int rows);
	MatrixXd feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias);
	void evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_out, MatrixXd target_class, MatrixXd bias);
};

const double cw_act(double x);
const double cw_act_deriv(double x);
const double cw_calc_err(MatrixXd input, MatrixXd target);