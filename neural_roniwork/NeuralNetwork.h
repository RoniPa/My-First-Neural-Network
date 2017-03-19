#pragma once

#include <Eigen/Dense>

namespace nrt {
	using namespace Eigen;

	class NeuralNetwork {
	public:
		NeuralNetwork(int nI, int nH, int nO);
		~NeuralNetwork();

		int inputs();
		int hiddens();
		int outputs();

		void feedForward(double* pattern);
		void backpropagate();
	private:
		int m_input;
		int m_hidden;
		int m_output;

		double* m_inputNeurons;
		double* m_hiddenNeurons;
		double* m_outputNeurons;

		double activate(double x);
		double activate_deriv(double x);

		unsigned int output_to_class(double* output);
		double* class_to_output(int x);
	};
}