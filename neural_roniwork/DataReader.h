#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Types.h"

namespace nrt {
	using namespace Eigen;

	class DataReader {
	public:
		DataSet readData(std::string train_uri,std::string val_uri,std::string test_uri);
	private:
		SingleSet readSingle(std::string f_uri);
	};

	template<typename T = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
	MatrixXd convert_to_matrix(int cols, int rows, std::vector<std::vector<T>> raw_data);

	template<typename T = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
	MatrixXd convert_to_matrix(int cols, int rows, std::vector<T> raw_data);
}
