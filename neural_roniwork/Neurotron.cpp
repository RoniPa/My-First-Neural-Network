#include "Neurotron.h"
#include "NeurotronDataset.h"
#include <math.h>
#include <iostream>

namespace nrt {

	Neurotron::Neurotron() {}

	void Neurotron::init(int rows, int cols) 
	{
		m_init_weight(1, rows, cols);
		m_bias = MatrixXd::Constant(1, cols, 1);
		std::cout << "weights=" << std::endl << m_weight << std::endl
			<< "bias=" << std::endl << m_bias;

		NeurotronDataset dr;
		MatrixXd data = dr.read_data("iris_training.dat");
		std::cout << data;
	}

	MatrixXd Neurotron::m_activate(MatrixXd m) 
	{
		return m.unaryExpr(&cw_act);
	}

	MatrixXd Neurotron::m_activate_d(MatrixXd m) 
	{
		return m.unaryExpr(&cw_act_deriv);
	}

	// Init weight matrix with random values between -1*max and max
	MatrixXd Neurotron::m_init_weight(double max, int rows, int cols) 
	{
		m_weight = MatrixXd::Random(rows, cols);
		m_weight = (m_weight + MatrixXd::Constant(rows, cols, 1.))*max; // add 1 to the matrix to have values between 0 and 2; multiply with max value
		m_weight = (m_weight + MatrixXd::Constant(rows, cols, -max));
		return m_weight;
	}

	// Return updated weight matrix
	MatrixXd Neurotron::m_backpropagate(MatrixXd input, double eta, MatrixXd bias) 
	{
		return m_weight;
	}

	MatrixXd Neurotron::feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias) 
	{
		MatrixXd hc(input.rows(), input.cols() + bias.cols());
		hc << input, bias;
		m_net = weight*hc;
		m_output = m_activate(m_net);
		return m_output;
	}

	void Neurotron::evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_out, MatrixXd target_class, MatrixXd bias) 
	{
		m_output = feed_forward(input, weight, bias);
		m_error = calc_err(m_output, target_out);
		m_class_match_count = count_match_err(m_output, target_class);
	}

	// ------------- Helper functions ------------- //

	inline const double cw_act(double x) // Componentwise init
	{
		return (tanh(x) + 1) / 2;
	}

	inline const double cw_act_deriv(double x) 
	{
		double t = tanh(x);
		return (1 - pow(tanh(x), 2)) / 2;
	}

	// Conversion from raw value vector to class
	inline const int rw_convert_to_class(MatrixXd output) 
	{
		int c = 0;
		for (int i = output.cols() - 1; i >= 0; i--) {
			c = c | ((int)output(0, i) << i);
		}
		std::cout << "\t" << c;
		return c;
	}

	// Conversion from class to raw value vector
	inline const MatrixXd rw_convert_from_class(int x) 
	{
		MatrixXd m(1, 3);
		for (int i = 2; i >= 0; i--) {
			m(0, i) = x >> 1;
		}
		return m;
	}

	inline const double calc_err(MatrixXd input, MatrixXd target) 
	{
		int input_size = (int)(input.rows() * input.cols());
		MatrixXd err = input - target;
		err *= err;
		return err.sum() / (pow(input_size, 2));
	}

	// Input as raw matrix, target as class matrix
	inline const int count_match_err(MatrixXd output, MatrixXd target_classes) 
	{
		int class_err_count = 0,
			row_count = (int)output.rows();

		for (int i = row_count - 1; i >= 0; i--) {
			class_err_count += (int)(rw_convert_to_class(output.row(i)) != target_classes(i, 0));
		}
		return class_err_count;
	}
}