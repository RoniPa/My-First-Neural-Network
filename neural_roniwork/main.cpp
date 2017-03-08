#include <Eigen/Dense>
#include "Neurotron.h"
#include "NeurotronDataset.h"

#include <iostream>

using namespace nrt;

int main() {
	Neurotron n;
	n.init(3, 3);

	MatrixXd testi1(5, 3);
	MatrixXd testi2(5, 1);

	testi1(0, 0) = 0;
	testi1(0, 1) = 0;
	testi1(0, 2) = 1;
	testi1(1, 0) = 1;
	testi1(1, 1) = 0;
	testi1(1, 2) = 0;
	testi1(2, 0) = 0;
	testi1(2, 1) = 1;
	testi1(2, 2) = 0;
	testi1(3, 0) = 1;
	testi1(3, 1) = 0;
	testi1(3, 2) = 0;
	testi1(4, 0) = 0;
	testi1(4, 1) = 0;
	testi1(4, 2) = 1;

	testi2(0, 0) = 4;
	testi2(1, 0) = 1;
	testi2(2, 0) = 2;
	testi2(3, 0) = 1;
	testi2(4, 0) = 4;
	
	std::cout << "\nMatches: " << count_match_err(testi1, testi2);

	std::cin.get();
	return 0;
}