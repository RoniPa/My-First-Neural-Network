#include "NeurotronDataset.h"
#include <iostream>

namespace nrt {

	MatrixXd NeurotronDataset::m_convert_to_matrix(int cols, int rows, std::vector<std::vector<double>> raw_data) {
		MatrixXd output(rows, cols);

		for (std::vector<std::vector<double>>::const_iterator i = raw_data.begin(); i != raw_data.end(); ++i) {
			int curr_row = i - raw_data.begin();
			for (std::vector<double>::const_iterator j = i->begin(); j != i->end(); ++j)
				output(curr_row, j - i->begin()) = *j;
		}

		return output;
	}

	MatrixXd NeurotronDataset::read_data(std::string f_uri) {
		std::vector<std::vector<double>> training_data;
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
						value = Neurotron::SETOSA;
					}
					else if (s.compare("Iris-versicolor") == 0) {
						value = Neurotron::VERSICOLOR;
					}
					else if (s.compare("Iris-virginica") == 0) {
						value = Neurotron::VIRGINICA;
					}
				}
				record.push_back(value);
				++cols;
			}
			training_data.push_back(record);
			++rows;
		}
		cols /= rows;

		m_input_count = rows - 1;

		if (!infile.eof()) throw "Fooey!\n";

		infile.close();
		return m_convert_to_matrix(cols, rows, training_data);
	}
}