#pragma once

#include <Eigen/Dense>

namespace nrt {
	using namespace Eigen;

	struct NetworkError {
		double error;
		int classError;
	};

	struct TrainSet {
		MatrixXd values;
		MatrixXd targets;
	};
}