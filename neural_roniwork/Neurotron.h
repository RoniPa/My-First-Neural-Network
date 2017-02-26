#pragma once

#include <Eigen/Dense>

using namespace Eigen;

class Neurotron {
private:
	int			m_class_match_count;

	double		m_error;

	MatrixXd	m_net,
				m_weight,
				m_bias,
				m_output;

	MatrixXd	m_target_out, 
				m_target_class;

	MatrixXd m_activate(MatrixXd m);
	MatrixXd m_activate_d(MatrixXd m);
	MatrixXd m_init_weight(double max, int width, int height);
	MatrixXd m_backpropagate(MatrixXd input, double eta, MatrixXd bias);

public:
	const static int	SETOSA		= 0b001, // matches to [1,0,0]
						VERSICOLOR	= 0b010, // matches to [0,1,0]
						VIRGINICA	= 0b100; // matches to [0,0,1]

	Neurotron();
	void init(int cols, int rows);
	MatrixXd feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias);
	void evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_out, MatrixXd target_class, MatrixXd bias);
};


const double	cw_act(double x);
const double	cw_act_deriv(double x);

const int		rw_convert_to_class(MatrixXd x);
const MatrixXd	rw_convert_from_class(int x);

const double	calc_err(MatrixXd input, MatrixXd target);
const int		count_matches(MatrixXd output, MatrixXd target);