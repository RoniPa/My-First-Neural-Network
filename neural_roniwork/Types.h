#pragma once

#include <vector>

namespace nrt {

	struct SingleSet {
		unsigned int dataCount;
		std::vector<std::vector<double>> values;
		std::vector<std::vector<double>> targets;
	};

	struct DataSet {
		// Input count
		unsigned int inputCount;

		// Output class count
		unsigned int outputCount;

		// Output class values
		const static unsigned int SETOSA = 0b001; // matches to [0,0,1]
		const static unsigned int VERSICOLOR = 0b010; // matches to [0,1,0]
		const static unsigned int VIRGINICA = 0b100; // matches to [1,0,0]

		// Data sets
		SingleSet training;
		SingleSet validation;
		SingleSet test;
	};
}
