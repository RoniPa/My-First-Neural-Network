#include "Neurotron.h"
#include "NeurotronDataset.h"
#include <math.h>
#include <iostream>

#if defined(DEBUG) || defined(_DEBUG)
# define D(x) x
#else
# define D(x)
#endif // DEBUG

namespace nrt {

	Neurotron::Neurotron() {}

	// Train network with given data sets
	//
	// @param	tr_data		Training data set
	// @param	vd_data		Validation data set
	// @param	ts_data		Test data set
	//
	MatrixXd Neurotron::train(TrainSet tr_data, TrainSet vd_data, TrainSet ts_data)
	{
		m_init_weight(0.5, tr_data.values.rows() + 1, tr_data.values.cols());
		m_init_biases(tr_data.values, vd_data.values, ts_data.values);

		D(
			double tr_err[500];
			double tr_class_err[500];
			double vd_err[500];
			double vd_class_err[500];
			double ts_err[500];
			double ts_class_err[500];
		)

		// Iterate
		int i = 0;
		while (i < 500) {
			m_weight = m_backpropagate(tr_data.values, 0.1, m_biases[0]);
			
			D(
				NetworkError err1 = m_evaluate(tr_data.values, m_weight, tr_data.targets, m_biases[0]);
				NetworkError err2 = m_evaluate(vd_data.values, m_weight, vd_data.targets, m_biases[1]);
				NetworkError err3 = m_evaluate(ts_data.values, m_weight, ts_data.targets, m_biases[2]);

				tr_err[i] = err1.error;
				tr_class_err[i] = err1.classError;
				vd_err[i] = err2.error;
				vd_class_err[i] = err2.classError;
				ts_err[i] = err3.error;
				ts_class_err[i] = err3.classError;
			)

			i++;
		}

		D(
			NetworkError err1 = m_evaluate(tr_data.values, m_weight, tr_data.targets, m_biases[0]);
			NetworkError err2 = m_evaluate(vd_data.values, m_weight, vd_data.targets, m_biases[1]);
			NetworkError err3 = m_evaluate(ts_data.values, m_weight, ts_data.targets, m_biases[2]);

			tr_err[i] = err1.error;
			tr_class_err[i] = err1.classError;
			vd_err[i] = err2.error;
			vd_class_err[i] = err2.classError;
			ts_err[i] = err3.error;
			ts_class_err[i] = err3.classError;
		)

		for (int i = 0; i < 500; i++) {
			std::cout << tr_err[i];
		}

		return m_weight;
	}

	MatrixXd* Neurotron::m_init_biases(MatrixXd tr_data, MatrixXd vd_data, MatrixXd ts_data)
	{	
		m_biases[0] = MatrixXd::Constant(tr_data.rows(), 1, 1);	// training
		m_biases[1] = MatrixXd::Constant(vd_data.rows(), 1, 1); // validation
		m_biases[2] = MatrixXd::Constant(ts_data.rows(), 1, 1); // test
		return m_biases;
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

	// Backpropagation
	//
	// @param	input		Input data matrix (MatrixXd)
	// @param	eta			Learning rate
	// @param	bias		Bias matrix (MatrixXd)
	//
	// @return	Updated weight matrix
	//
	MatrixXd Neurotron::m_backpropagate(MatrixXd input, double eta, MatrixXd bias) 
	{
		return m_weight;
	}

	MatrixXd Neurotron::m_feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias)
	{
		MatrixXd hc(input.rows(), input.cols() + bias.cols());
		hc << input, bias;
		//m_net = weight*hc; <-- todo: these can't be multiplied
		std::cout << m_activate(m_net);
		m_output = m_activate(m_net);
		return m_output;
	}

	NetworkError Neurotron::m_evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_class, MatrixXd bias)
	{
		NetworkError error;
		m_output = m_feed_forward(input, weight, bias);
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