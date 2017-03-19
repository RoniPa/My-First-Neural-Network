#include "DataReader.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

using namespace nrt;

int main() {
	DataReader dr;
	DataSet data = dr.readData(
		"iris_training.dat", "iris_validation.dat", "iris_test.dat");

	NeuralNetwork nn(4, 4, 3);
	NeuralNetworkTrainer nT(&nn);

	std::cout << data.training.values << data.training.targets;

	std::cin.get();
	return 0;
}