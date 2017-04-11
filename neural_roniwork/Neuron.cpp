#include "Neuron.h"

#define LEARNING_RATE 0.08
// #define LEARNING_RATE 0.015
#define MOMENTUM 0.5

namespace nrt {

	const double Neuron::eta = LEARNING_RATE;
	const double Neuron::alpha = MOMENTUM;

	void Neuron::updateInputWeights(Layer &prevLayer)
	{
		// The weights to be updated are in the Connection container
		// in the neurons in the preceding layer
		for (unsigned n = 0; n < prevLayer.size(); ++n) {
			Neuron &neuron = prevLayer[n];
			double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

			double newDeltaWeight =
				// Individual input, magnified by the gradient and learning rate:
				Neuron::eta * neuron.getOutputVal() * m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ Neuron::alpha * oldDeltaWeight;

			neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
		}
	}

	double Neuron::sumDOW(const Layer &nextLayer) const
	{
		double sum = 0.0;

		// Sum our contributions of the errors at the nodes we feed
		for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
			sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
		}

		return sum;
	}

	void Neuron::calcHiddenGradients(const Layer &nextLayer)
	{
		double dow = sumDOW(nextLayer);
		m_gradient = dow * Neuron::activationFuncDerivative(m_outputVal);
	}

	// Use mean squared error to count the cost
	void Neuron::calcOutputGradients(double targetVal)
	{
		double delta = targetVal - m_outputVal;
		m_gradient = delta * Neuron::activationFuncDerivative(m_outputVal);
	}

	double Neuron::activationFunc(double x)
	{
		// hyperbolic tangent - output range [-1.0 ... 1.0]
		return floor(tanh(x) * 10000 + 0.5) / 10000;
	}

	double Neuron::activationFuncDerivative(double x)
	{
		// tanh derivative approximation
		return 1.0 - x * x;
	}

	void Neuron::activate(const Layer &prevLayer)
	{
		double sum = 0.0;
		for (unsigned n = 0; n < prevLayer.size(); ++n) {
			sum += prevLayer[n].getOutputVal() *
				prevLayer[n].m_outputWeights[m_myIndex].weight;
		}

		m_outputVal = Neuron::activationFunc(sum);
	}

	// Constructor
	Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
	{
		for (unsigned c = 0; c < numOutputs; ++c) {
			m_outputWeights.push_back(Connection());
			m_outputWeights.back().weight = randomWeight();
		}

		m_myIndex = myIndex;
	}
}