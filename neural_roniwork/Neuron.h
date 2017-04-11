#pragma once

#include <vector>
#include <cmath>

namespace nrt {

	struct Connection
	{
		double weight;
		double deltaWeight;
	};

	class Neuron;
	typedef std::vector<Neuron> Layer;

	class Neuron
	{
	public:
		static const double eta;	// Overall NeuralNetwork training rate [0.0 ... 1.0]
		static const double alpha;	// Multiplier of last weight change (momentum) [0.0 ... n]
		Neuron(unsigned numOutputs, unsigned myIndex);
		void setOutputVal(double val) { m_outputVal = val; };
		double getOutputVal(void) const { return m_outputVal; };
		void activate(const Layer &prevLayer);
		void calcOutputGradients(double targetVal);
		void calcHiddenGradients(const Layer &nextLayer);
		void updateInputWeights(Layer &prevLayer);
		std::vector<Connection> getWeights() const { return m_outputWeights; };

	private:
		static double activationFunc(double x);
		static double activationFuncDerivative(double x);
		static double randomWeight(void) { return rand() / (double(RAND_MAX)*2); } // [-0.5 ... 0.5]
		double sumDOW(const Layer &nextLayer) const;
		double m_outputVal;
		unsigned m_myIndex; // Neuron index in layer
		double m_gradient;
		std::vector<Connection> m_outputWeights; // Weights to next layer neurons from this neuron
	};
}