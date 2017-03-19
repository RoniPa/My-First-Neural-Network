#include <iostream>

#include "NeurotronTrainer.h";

namespace nrt {
	NeurotronTrainer::NeurotronTrainer(Neurotron n) : m_net{ n } 
	{}

	// Train network with given data sets
	//
	// @param	tr_data		Training data set
	// @param	vd_data		Validation data set
	// @param	ts_data		Test data set
	//
	Neurotron NeurotronTrainer::train(TrainSet tr_data, TrainSet vd_data, TrainSet ts_data)
	{
			m_net.init_weights(0.5, tr_data.values.rows() + 1, tr_data.values.cols());
			m_net.init_biases(tr_data.values, vd_data.values, ts_data.values);

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
				m_net.weights = m_net.backpropagate(tr_data.values, 0.1, m_net.biases[0]);

				D(
					NetworkError err1 = m_net.evaluate(tr_data.values, m_net.weights, tr_data.targets, m_net.biases[0]);
					NetworkError err2 = m_net.evaluate(vd_data.values, m_net.weights, vd_data.targets, m_net.biases[1]);
					NetworkError err3 = m_net.evaluate(ts_data.values, m_net.weights, ts_data.targets, m_net.biases[2]);

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
				NetworkError err1 = m_net.evaluate(tr_data.values, m_net.weights, tr_data.targets, m_net.biases[0]);
				NetworkError err2 = m_net.evaluate(vd_data.values, m_net.weights, vd_data.targets, m_net.biases[1]);
				NetworkError err3 = m_net.evaluate(ts_data.values, m_net.weights, ts_data.targets, m_net.biases[2]);

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

			return m_net;
	}
}