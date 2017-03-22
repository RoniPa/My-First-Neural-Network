#include "NeuralNetworkTrainer.h"
#include "DataReader.h"

namespace nrt {

	NeuralNetworkTrainer::NeuralNetworkTrainer(NeuralNetwork* nn) :
		m_nn(nn),
		m_maxEpochs(MAX_EPOCHS),
		m_desiredAccuracy(DESIRED_ACCURACY)
	{

	}

	void NeuralNetworkTrainer::trainNeuralNetwork(DataSet* tData)
	{
		std::cout << std::endl << " Neural Network Training Starting: " << std::endl
			<< "==========================================================================" << std::endl
			<< " LR: " << Neuron::eta << ", Momentum: " << Neuron::alpha << ", Max Epochs: " << m_maxEpochs << std::endl
			<< "==========================================================================" << std::endl << std::endl;

		m_epoch = 0;
		
		while (m_epoch < 100) { ++m_epoch;
			m_runTrainingEpoch(tData->training);
		}
	}

	void NeuralNetworkTrainer::m_runTrainingEpoch(const SingleSet &eSet)
	{
		int err_count = 0;

		std::cout << "Epoch " << m_epoch << std::endl;

		std::vector<double> resultVals(3);
		for (unsigned i = 0; i < eSet.dataCount; ++i) {
			std::vector<double> inputVals {
				eSet.values(i, 0),
				eSet.values(i, 1),
				eSet.values(i, 2),
				eSet.values(i, 3)
			};
			m_nn->feedForward(inputVals);

			std::vector<double> targetVals {
				eSet.targets(i, 0),
				eSet.targets(i, 1),
				eSet.targets(i, 2)
			};
			m_nn->backProp(targetVals);
			m_nn->getResults(resultVals);

			int targetClass = DataReader::convert_to_class(targetVals);
			int result = DataReader::convert_to_class(resultVals);
			if (targetClass != result) ++err_count;
			
			//std::cout << "target: " << targetClass << ", result: " << result << "\n";
		}
		
		std::cout << "AVG. NET DIFF: " << m_nn->getError() 
			<< " CLASS VALIDATION ERRORS: " 
			<< (int)((double)err_count/(double)eSet.dataCount*100) << " %" << std::endl;
	}
}