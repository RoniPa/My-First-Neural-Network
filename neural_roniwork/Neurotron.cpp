#include "Neurotron.h"
#include <math.h>
#include <iostream>

Neurotron::Neurotron() {}

void Neurotron::init(int rows, int cols) {
	m_init_weight(1, rows, cols);
	m_bias = MatrixXd::Constant(1, cols, 1);
	std::cout << "weights=" << std::endl << m_weight << std::endl
		<< "bias=" << std::endl << m_bias;
}

MatrixXd Neurotron::m_activate(MatrixXd m) {
	return m.unaryExpr(&cw_act);
}

MatrixXd Neurotron::m_activate_d(MatrixXd m) {
	return m.unaryExpr(&cw_act_deriv);
}

MatrixXd Neurotron::feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias) {
	MatrixXd hc(input.rows(), input.cols() + bias.cols());
	hc << input, bias;
	m_net = weight*hc;
	m_output = m_activate(m_net);
	return m_output;
}

void Neurotron::evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_out, MatrixXd target_class, MatrixXd bias) {
	m_output = feed_forward(input, weight, bias);
	m_error = cw_calc_err(m_output, target_out);
	m_class_match_count = calc_matches(m_output, target_out);
}

MatrixXd Neurotron::m_init_weight(double max, int rows, int cols) {
	m_weight = MatrixXd::Random(rows, cols);
	m_weight = (m_weight + MatrixXd::Constant(rows, cols, 1.))*max; // add 1 to the matrix to have values between 0 and 2; multiply with max value
	m_weight = (m_weight + MatrixXd::Constant(rows, cols, -max));
	return m_weight;
}

const double cw_act(double x) { // Componentwise init
	return (tanh(x) + 1) / 2;
}

const double cw_act_deriv(double x) {
	double t = tanh(x);
	return (1 - pow(tanh(x), 2)) / 2;
}

const double cw_calc_err(MatrixXd input, MatrixXd target) {
	int input_size = input.rows() * input.cols();
	MatrixXd err = input - target;
	err *= err;
	return err.sum() / (pow(input_size, 2));
}

const double calc_matches(MatrixXd input, MatrixXd target) {
	double class_err_count = 1;
	// todo: Calculate & normalize class matches
	return class_err_count;
}