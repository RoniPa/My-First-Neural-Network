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

		int initial_err_count = m_runValidation(tData->test, false);
		double initial_error = m_nn->getError();

		int errCount1, errCount2;
		double accuracy1, accuracy2;
		double tSize = (double)tData->training.dataCount;
		double vSize = (double)tData->validation.dataCount;
		unsigned max_validation_err = (100 - DESIRED_ACCURACY);

		do { ++m_epoch;
			std::cout << "-----------------" << std::endl;
			std::cout << "EPOCH " << m_epoch << std::endl;
			std::cout << "-----------------" << std::endl;

			errCount1 = m_runTraining(tData->training);
			errCount2 = m_runValidation(tData->validation, false);

			accuracy1 = ((double)errCount1 / tSize) * 100;
			accuracy2 = ((double)errCount2 / vSize) * 100;
		} while (
			m_epoch < MAX_EPOCHS 
			&& m_nn->getError() > DESIRED_MSE 
			&& (accuracy1 > max_validation_err
			|| accuracy2 > max_validation_err)
		);

		std::cout << std::endl << "=========================================================================="
			<< std::endl << " Neural Network Training Finished " << std::endl
			<< "==========================================================================" << std::endl << std::endl;

		// Training result output

		std::cout << "\n\nInitial results with validation data\n------------------------------\n";
		std::cout << "AVG. NET DIFF: " << initial_error
			<< " CLASS VALIDATION ERRORS: "
			<< (int)((double)initial_err_count / (double)(tData->test).dataCount * 100) << " %, "
			<< initial_err_count << " ptc" << std::endl;
				
		
		m_runValidation(tData->test);

		std::cout << std::endl << std::endl;
		std::cout << "Result network weights by neuron:\n\n";
		std::vector<std::vector<double>> netWeights = m_nn->getWeights();
		for (std::vector<std::vector<double>>::const_iterator i = netWeights.begin(); i != netWeights.end(); ++i) {
			int curr_row = (int)(i - netWeights.begin());
			for (std::vector<double>::const_iterator j = i->begin(); j != i->end(); ++j)
				std::cout << std::setw(12) << *j << "\t";
			std::cout << std::endl;
		}
	}

	int myrandom(int i) { return std::rand() % i; }
	int NeuralNetworkTrainer::m_runTraining(const SingleSet &eSet)
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

				std::cout << "Target: " << targetClass << " | Result: " << result;
				std::cout << " | Output probabilities:"
					<< std::setw(7) << resultVals[0] << "\t"
					<< std::setw(7) << resultVals[1] << "\t"
					<< std::setw(7) << resultVals[2] << std::endl;
			}
		}
		
		std::cout << "AVG. NET DIFF: " << m_nn->getError() 
			<< " CLASS VALIDATION ERRORS: " 
			<< (int)((double)err_count/(double)eSet.dataCount*100) << " %" << std::endl;

		return err_count;
	}

	int NeuralNetworkTrainer::m_runValidation(const SingleSet &eSet, bool verbose)
	{
		int err_count = 0;
		std::vector<double> resultVals(3);

		if (verbose) {
			std::cout << "\n\nResults with validation data\n------------------------------\n";
		}

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

			if (verbose) {
				std::bitset<3> x(targetClass);
				std::cout << "Target: " << x << " | Output probabilities: "
					<< std::setw(7) << resultVals[0] << "\t"
					<< std::setw(7) << resultVals[1] << "\t"
					<< std::setw(7) << resultVals[2] << std::endl;
			}
		}

		if (verbose) {
			std::cout << "AVG. NET DIFF: " << m_nn->getError()
				<< " CLASS VALIDATION ERRORS: "
				<< (int)((double)err_count / (double)eSet.dataCount * 100) << " %, "
				<< err_count << " ptc" << std::endl;
		}

		return err_count;
	}
}