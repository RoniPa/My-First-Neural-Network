#include <Eigen/Dense>
#include "Neurotron.h"
#include "NeurotronDataset.h"

#include <iostream>

using namespace nrt;

int main() {

	Neurotron n;
	NeurotronDataset dr;
	TrainSet tr_data = dr.read_data("iris_training.dat");
	TrainSet vd_data = dr.read_data("iris_validation.dat");
	TrainSet ts_data = dr.read_data("iris_test.dat");
	n.train(tr_data, vd_data, ts_data);

	std::cin.get();
	return 0;
}