#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Types.h"

namespace nrt {

	class DataReader {
	public:
		DataSet readData(std::string train_uri,std::string val_uri,std::string test_uri);
		static int DataReader::convert_to_class(const std::vector<double> output);
		static std::vector<double> DataReader::convert_from_class(unsigned x);
	private:
		SingleSet readSingle(std::string f_uri);
	};

}
