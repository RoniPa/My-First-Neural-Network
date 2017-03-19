#include "NeuralNetworkTrainer.h"

namespace nrt {

	NeuralNetworkTrainer::NeuralNetworkTrainer(NeuralNetwork* nn) :
		m_nn(nn),
		m_learningRate(LEARNING_RATE),
		m_momentum(MOMENTUM),
		m_maxEpochs(MAX_EPOCHS),
		m_desiredAccuracy(DESIRED_ACCURACY)
	{

	}

	void NeuralNetworkTrainer::trainNetwork(DataSet* tData)
	{
		std::cout << std::endl << " Neural Network Training Starting: " << std::endl
			<< "==========================================================================" << std::endl
			<< " LR: " << m_learningRate << ", Momentum: " << m_momentum << ", Max Epochs: " << m_maxEpochs << std::endl
			<< " " << m_nn->inputs() << " Input Neurons, " << m_nn->hiddens() << " Hidden Neurons, " << m_nn->outputs() << " Output Neurons" << std::endl
			<< "==========================================================================" << std::endl << std::endl;

		m_epoch = 0;

		// todo: Replace with accuracy check
		while (m_epoch < 100) { ++m_epoch;
			m_runTrainingEpoch(tData->training);
		}
	}

	void NeuralNetworkTrainer::m_runTrainingEpoch(SingleSet eSet)
	{
		std::cout << "Epoch " << m_epoch;
	}
}