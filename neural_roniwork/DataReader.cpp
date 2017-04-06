#include "DataReader.h"

namespace nrt {
	
	DataSet DataReader::readData(std::string train_uri,std::string val_uri,std::string test_uri)
	{
		DataSet result;
		result.training = readSingle(train_uri);
		result.validation = readSingle(val_uri);
		result.test = readSingle(test_uri);

		result.inputCount = 4;
		result.outputCount = 3;
		return result;
	}

	SingleSet DataReader::readSingle(std::string f_uri)
	{
		std::vector<std::vector<double>> data;
		std::vector<std::vector<double>> target;

		std::ifstream infile(f_uri);

		int rows(0), cols(0);
		while (infile)
		{
			std::string s;
			if (!getline(infile, s)) break;

			std::istringstream ss(s);
			std::vector<double> record;

			while (ss)
			{
				std::string s;
				std::stringstream convert;
				double value;

				if (!getline(ss, s, ',')) break;
				convert = std::stringstream(s);
				if (!(convert >> value)) {
					if (s.compare("Iris-setosa") == 0) {
						value = DataSet::SETOSA;
					}
					else if (s.compare("Iris-versicolor") == 0) {
						value = DataSet::VERSICOLOR;
					}
					else if (s.compare("Iris-virginica") == 0) {
						value = DataSet::VIRGINICA;
					}
					target.push_back(convert_from_class(value));
				}
				else {
					record.push_back(value);
					++cols;
				}
			}
			data.push_back(record);
			++rows;
		}
		cols /= rows;

		if (!infile.eof()) throw "Fooey!\n";

		infile.close();

		SingleSet result;
		result.dataCount = rows - 1;
		result.values = data;
		result.targets = target;
		return result;
	}

	// Conversion from raw value vector to class
	int DataReader::convert_to_class(const std::vector<double> output)
	{
		double curr_max = 0;
		int max_ind = 0;
		for (int i = output.size() - 1; i >= 0; i--) {
			if (curr_max < output[i]) {
				curr_max = output[i];
				max_ind = i;
			}
		}

		return max_ind == 2 ? 0b001 : max_ind == 1 ? 0b010 : 0b100;
	}

	// Conversion from class to raw value vector
	std::vector<double> DataReader::convert_from_class(unsigned x)
	{
		std::vector<double> m;
		for (int i = 2; i >= 0; i--) {
			m.push_back((x >> i) & 0b001);
		}
		return m;
	}
}