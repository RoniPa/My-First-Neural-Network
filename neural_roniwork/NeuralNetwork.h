#pragma once

#include <vector>
#include <iostream>
#include <cassert>
#include "Neuron.h"

namespace nrt {

	class NeuralNetwork
	{
	public:
		NeuralNetwork(const std::vector<unsigned> &topology);
		void feedForward(const std::vector<double> &inputVals);
		void backProp(const std::vector<double> &targetVals);
		void getResults(std::vector<double> &resultVals) const;
		double getError() const { return m_error; };
		std::vector<std::vector<double>> getWeights() const;

	private:
		std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
	};

}