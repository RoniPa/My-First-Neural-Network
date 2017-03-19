#include "Neurotron.h"
#include "NeurotronDataset.h"
#include <math.h>
#include <iostream>

namespace nrt {

	Neurotron::Neurotron() 
	{
		// ...
	}

	Neurotron::~Neurotron()
	{
		// ...
	}

	MatrixXd* Neurotron::init_biases(MatrixXd tr_data, MatrixXd vd_data, MatrixXd ts_data)
	{	
		biases[0] = MatrixXd::Constant(tr_data.rows(), 1, 1); // training
		biases[1] = MatrixXd::Constant(vd_data.rows(), 1, 1); // validation
		biases[2] = MatrixXd::Constant(ts_data.rows(), 1, 1); // test
		return biases;
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
	MatrixXd Neurotron::init_weights(double max, int rows, int cols) 
	{
		weights = MatrixXd::Random(rows, cols);
		weights = (weights + MatrixXd::Constant(rows, cols, 1.))*max; // add 1 to the matrix to have values between 0 and 2; multiply with max value
		weights = (weights + MatrixXd::Constant(rows, cols, -max));
		return weights;
	}

	// Backpropagation
	//
	// @param	input		Input data matrix (MatrixXd)
	// @param	eta			Learning rate
	// @param	bias		Bias matrix (MatrixXd)
	//
	// @return	Updated weight matrix
	//
	MatrixXd Neurotron::backpropagate(MatrixXd input, double eta, MatrixXd bias) 
	{
		// todo: Implement backpropagation
		return weights;
	}

	MatrixXd Neurotron::m_feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias)
	{
		MatrixXd hc(input.rows(), input.cols() + 1);
		hc << input, bias;
		m_net = weight*hc;
		std::cout << hc << "\n\n" << weights;
		std::cout << m_activate(m_net);
		m_output = m_activate(m_net);
		return m_output;
	}

	NetworkError Neurotron::evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_class, MatrixXd bias)
	{
		int rowCount = input.rows();
		NetworkError error;
		
		for (int i = 0; i < rowCount; i++) {
			m_output.row(i) = m_feed_forward(input.row(i), weight.row(0), bias);
		}
		error.error = calc_err(m_output, target_class);
		error.classError = count_match_err(m_output, target_class);
		return error;
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

	inline const double calc_err(MatrixXd input, MatrixXd target_classes) 
	{
		int input_size = (int)(input.rows() * input.cols());
		int row_count = (int)target_classes.rows();
		MatrixXd target(row_count, 3);

		for (int i = row_count - 1; i >= 0; i--) {
			target.row(i) = rw_convert_from_class(target_classes(i, 0));
		}

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