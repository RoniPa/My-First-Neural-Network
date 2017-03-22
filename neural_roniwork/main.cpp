#include <vector>
#include <iostream>

#include "DataReader.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

using namespace nrt;

int main() {
	DataReader dr;
	DataSet data = dr.readData(
		"iris_training.dat", "iris_validation.dat", "iris_test.dat");

	std::vector<unsigned> netTopology {4, 4, 3};
	NeuralNetwork nn(netTopology);
	NeuralNetworkTrainer nT(&nn);
	nT.trainNeuralNetwork(&data);

	std::cin.get();
	return 0;
}