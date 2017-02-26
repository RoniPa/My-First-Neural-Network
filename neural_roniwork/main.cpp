#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "Neurotron.h"

int main() {
	std::vector <std::vector<double>> training_data;
	MatrixXd inputdata;
	std::ifstream infile("iris_training.dat");

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
				} else if (s.compare("Iris-versicolor") == 0) {
					value = Neurotron::VERSICOLOR;
				} else if (s.compare("Iris-virginica") == 0) {
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

	if (!infile.eof())
	{
		std::cerr << "Fooey!\n";
	}

	infile.close();
	
	for (std::vector<std::vector<double>>::const_iterator i = training_data.begin(); i != training_data.end(); ++i) {
		for (std::vector<double>::const_iterator j = i->begin(); j != i->end(); ++j)
			std::cout << *j << '\t';
		std::cout << std::endl;
	}
		

	Neurotron n;
	n.init(3, 3);

	std::cin.get();
	return 0;
}