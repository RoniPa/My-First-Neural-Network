#pragma once

#include "Types.h"
#include "NeuralNetwork.h"
#include <iostream>

// Set defaults
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 90  
#define DESIRED_MSE 0.001 

namespace nrt {
	class NeuralNetworkTrainer {
	public:
		NeuralNetworkTrainer(NeuralNetwork* nn);
		void trainNetwork(DataSet* tSet);
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

		void m_runTrainingEpoch(SingleSet eSet);
	};
}