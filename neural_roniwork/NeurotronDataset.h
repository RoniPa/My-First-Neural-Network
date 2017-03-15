#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Types.h"
#include "Neurotron.h"

namespace nrt {

	class NeurotronDataset {
	private:
		int m_input_count; // counted from data
		int m_output_count{ 3 };
	public:
		TrainSet read_data(std::string f_uri);
	};

	template<typename T = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
	MatrixXd convert_to_matrix(int cols, int rows, std::vector<std::vector<T>> raw_data);

	template<typename T = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
	MatrixXd convert_to_matrix(int cols, int rows, std::vector<T> raw_data);
}
