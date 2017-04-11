#pragma once

#include "Types.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <bitset>
#include <random>

// Set defaults
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 95  
#define DESIRED_MSE 0.008

namespace nrt {
	class NeuralNetworkTrainer {
	public:
		NeuralNetworkTrainer(NeuralNetwork* nn);
		void trainNeuralNetwork(DataSet* tSet);
	private:
		NeuralNetwork* m_nn;
		DataSet* m_tSet;

		//learning parameters
		double m_learningRate; // adjusts the step size of the weight update	
		double m_momentum; // improves performance of stochastic learning (don't use for batch)

		//epoch counter
		long m_epoch;
		long m_maxEpochs;

		//accuracy/MSE required
		double m_desiredAccuracy;

		int m_runTraining(const SingleSet &eSet);
		int m_runValidation(const SingleSet &eSet, bool verbose = true);
	};
}