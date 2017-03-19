#include "NeuralNetwork.h"
#include <algorithm>

namespace nrt {
	// Constructor
	NeuralNetwork::NeuralNetwork(int nI, int nH, int nO) :
		m_input{ nI },
		m_hidden{ nH },
		m_output{ nO }
	{
		// Initialize values to 0, 
		// add bias column to end with value 1

		m_inputNeurons = new double[m_input + 1];
		std::fill_n(m_inputNeurons, m_input, 0);
		m_inputNeurons[m_input] = 1;

		m_hiddenNeurons = new double[m_hidden + 1];
		std::fill_n(m_hiddenNeurons, m_hidden, 0);
		m_inputNeurons[m_hidden] = 1;

		m_outputNeurons = new double[m_output + 1];
		std::fill_n(m_outputNeurons, m_output, 0);
		m_inputNeurons[m_output] = 1;
	}

	// Destructor
	NeuralNetwork::~NeuralNetwork() 
	{
		delete[] m_inputNeurons;
		delete[] m_hiddenNeurons;
		delete[] m_outputNeurons;
	}

	int NeuralNetwork::inputs() { return m_input; }
	int NeuralNetwork::hiddens() { return m_hidden; }
	int NeuralNetwork::outputs() { return m_output; }

	void NeuralNetwork::feedForward(double* pattern)
	{

	}

	void NeuralNetwork::backpropagate()
	{

	}

	// Neutron activation function
	double NeuralNetwork::activate(double x)
	{
		// hyperbolic tangent function
		return (tanh(x) + 1) / 2;
	}

	double NeuralNetwork::activate_deriv(double x)
	{
		return (1 - pow(tanh(x), 2)) / 2;
	}

	// Conversion from raw value vector to class
	unsigned int NeuralNetwork::output_to_class(double* output)
	{
		int c = 0;
		for (int i = 2; i >= 0; i--) {
			c = c | ((int)output[i] << i);
		}
		return c;
	}

	// Conversion from class to raw value vector
	double* NeuralNetwork::class_to_output(int x)
	{
		double* output = new double[3];
		for (int i = 2; i >= 0; i--) {
			output[i] = x >> 1;
		}
		return output;
	}
}