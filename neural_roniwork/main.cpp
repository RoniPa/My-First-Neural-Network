#include <iostream>
#include <Eigen/Dense>
#include "Neurotron.h"

using namespace Eigen;

int main() {
	Neurotron n;
	
	std::cout << n.init_weight(1, 3, 3);
	std::cin.get();
	return 0;
}