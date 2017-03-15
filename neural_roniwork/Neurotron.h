#pragma once

#include "Types.h"

namespace nrt {
	using namespace Eigen;

	class Neurotron {
	public:
		const static int SETOSA = 0b001; // matches to [1,0,0]
		const static int VERSICOLOR = 0b010; // matches to [0,1,0]
		const static int VIRGINICA = 0b100; // matches to [0,0,1]

		Neurotron();
		MatrixXd train(TrainSet tr_data, TrainSet vd_data, TrainSet ts_data);
	private:
		MatrixXd m_biases[3];
		MatrixXd m_weight;

		MatrixXd m_net;
		MatrixXd m_output;
		MatrixXd m_target_out;
		MatrixXd m_target_class;
		
		MatrixXd m_feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias);
		NetworkError m_evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_class, MatrixXd bias);

		MatrixXd m_activate(MatrixXd m);
		MatrixXd m_activate_d(MatrixXd m);
		MatrixXd m_backpropagate(MatrixXd input, double eta, MatrixXd bias);
		MatrixXd m_init_weight(double max, int width, int height);
		MatrixXd* m_init_biases(MatrixXd tr_data, MatrixXd vd_data, MatrixXd ts_data);
	};

	const double	cw_act(double x);
	const double	cw_act_deriv(double x);

	const int		rw_convert_to_class(MatrixXd x);
	const MatrixXd	rw_convert_from_class(int x);

	const double	calc_err(MatrixXd input, MatrixXd target);
	const int		count_match_err(MatrixXd output, MatrixXd target);
}