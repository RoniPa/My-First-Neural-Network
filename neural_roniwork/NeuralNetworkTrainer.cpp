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

		int initial_err_count = m_runValidationEpoch(tData->test);
		double initial_error = m_nn->getError();

		int err1, err2;
		double accuracy;
		do { ++m_epoch;
			std::cout << "Epoch " << m_epoch << std::endl;
			err1 = m_runTrainingEpoch(tData->training);
			err2 = m_runTrainingEpoch(tData->validation);

			accuracy = ((double)err1 + (double)err2) /
				((double)tData->training.dataCount + (double)tData->validation.dataCount)
				* 100;
		} while (
			m_epoch < MAX_EPOCHS 
			&& m_nn->getError() > DESIRED_MSE 
			&& accuracy < DESIRED_ACCURACY
		);

		std::cout << "\n\nINITIAL RESULTS WITH VALIDATION DATA\n------------------------------\n";
		std::cout << "AVG. NET DIFF: " << initial_error
			<< " CLASS VALIDATION ERRORS: "
			<< (int)((double)initial_err_count / (double)(tData->test).dataCount * 100) << " %, "
			<< initial_err_count << " ptc" << std::endl;
				
		int end_err_count = m_runValidationEpoch(tData->test);
		std::cout << "\n\nEND RESULTS WITH VALIDATION DATA\n------------------------------\n";
		std::cout << "AVG. NET DIFF: " << m_nn->getError()
			<< " CLASS VALIDATION ERRORS: "
			<< (int)((double)end_err_count / (double)(tData->test).dataCount * 100) << " %, "
			<< end_err_count << " ptc" << std::endl;
	}

	int myrandom(int i) { return std::rand() % i; }
	int NeuralNetworkTrainer::m_runTrainingEpoch(const SingleSet &eSet)
	{
		int err_count = 0;
		std::vector<double> resultVals(3);

		// Randomize data sets
		std::vector<int> indexes;
		indexes.reserve(eSet.values.size());
		for (int i = 0; i < eSet.values.size(); ++i)
			indexes.push_back(i);
		std::random_shuffle(indexes.begin(), indexes.end(), myrandom);

		for (std::vector<int>::iterator it1 = indexes.begin(); it1 != indexes.end(); ++it1) {
			m_nn->feedForward(eSet.values[*it1]);
			m_nn->backProp(eSet.targets[*it1]);
			m_nn->getResults(resultVals);

			int targetClass = DataReader::convert_to_class(eSet.targets[*it1]);
			int result = DataReader::convert_to_class(resultVals);
			if (targetClass != result) {
				++err_count;

				std::cout << "target: " << targetClass << ", result: " << result;
				std::cout << ", output values:"
					<< resultVals[0] << ", "
					<< resultVals[1] << ", "
					<< resultVals[2] << std::endl;
			}
		}
		
		std::cout << "AVG. NET DIFF: " << m_nn->getError() 
			<< " CLASS VALIDATION ERRORS: " 
			<< (int)((double)err_count/(double)eSet.dataCount*100) << " %" << std::endl;

		return err_count;
	}

	int NeuralNetworkTrainer::m_runValidationEpoch(const SingleSet &eSet)
	{
		int err_count = 0;
		std::vector<double> resultVals(3);

		// Randomize data sets
		std::vector<int> indexes;
		indexes.reserve(eSet.values.size());
		for (int i = 0; i < eSet.values.size(); ++i)
			indexes.push_back(i);
		std::random_shuffle(indexes.begin(), indexes.end(), myrandom);

		for (std::vector<int>::iterator it1 = indexes.begin(); it1 != indexes.end(); ++it1) {
			m_nn->feedForward(eSet.values[*it1]);
			m_nn->getResults(resultVals);

			int targetClass = DataReader::convert_to_class(eSet.targets[*it1]);
			int result = DataReader::convert_to_class(resultVals);
			if (targetClass != result) {
				++err_count;
			}
		}

		return err_count;
	}
}