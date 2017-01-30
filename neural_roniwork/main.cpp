#include <iostream>
#include <Eigen/Dense>
#include "Neurotron.h"

using namespace Eigen;

int main() {
	Neurotron n;
	n.init(3, 3);

	std::cin.get();
	return 0;
}