#pragma once

#include <Eigen/Dense>

using namespace Eigen;

class Neurotron {
private:
	const int black = 0b01;
	const int white = 0b10;
	MatrixXd net;
	MatrixXd output;
public:
	Neurotron();
	MatrixXd init(MatrixXd m);
	MatrixXd init_derivate(MatrixXd m);
	MatrixXd init_weight(double max, int width, int height);
	void feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias);
	void evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_out, MatrixXd target_class, MatrixXd bias);
};

const double cw_init(double x);
const double wp_init_derivate(double x);