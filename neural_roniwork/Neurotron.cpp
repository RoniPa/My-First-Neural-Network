#include "Neurotron.h"
#include <math.h>

Neurotron::Neurotron() {}

MatrixXd Neurotron::init(MatrixXd m) {
	return m.unaryExpr(&cw_init);
}

MatrixXd Neurotron::init_derivate(MatrixXd m) {
	return m.unaryExpr(&wp_init_derivate);
}

MatrixXd Neurotron::init_weight(double max, int width, int height) {
	MatrixXd w = MatrixXd::Random(width, height);
	w = (w + MatrixXd::Constant(width, height, 1.))*max; // add 1 to the matrix to have values between 0 and 2; multiply with max value
	w = (w + MatrixXd::Constant(width, height, -max));
	return w;
}

void Neurotron::feed_forward(MatrixXd input, MatrixXd weight, MatrixXd bias) {
	MatrixXd hc(input.rows(), input.cols() + bias.cols());
	hc << input, bias;
	net = weight*hc;
	output = init(net);
}

void Neurotron::evaluate(MatrixXd input, MatrixXd weight, MatrixXd target_out, MatrixXd target_class, MatrixXd bias) {

}

const double cw_init(double x) { // Applied componentwise to init matrix
	return (tanh(x) + 1) / 2;
}

const double wp_init_derivate(double x) {
	double t = tanh(x);
	return (1 - pow(tanh(x), 2)) / 2;
}