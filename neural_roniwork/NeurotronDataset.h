#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Neurotron.h"

namespace nrt {

	class NeurotronDataset {
	private:
		int m_input_count;
		int m_output_count{ 3 };
		MatrixXd m_convert_to_matrix(int cols, int rows, std::vector<std::vector<double>> raw_data);
	public:
		MatrixXd read_data(std::string f_uri);
	};
}
