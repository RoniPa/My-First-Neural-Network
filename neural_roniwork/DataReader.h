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
		static int DataReader::convert_to_class(const std::vector<double> output);
		static std::vector<double> DataReader::convert_from_class(unsigned x);
	private:
		SingleSet readSingle(std::string f_uri);
	};

	template<typename T = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
	MatrixXd convert_to_matrix(int cols, int rows, std::vector<std::vector<T>> raw_data);
}
