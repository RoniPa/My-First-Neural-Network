#include "NeuralNetwork.h"

#define BIAS -1.0

namespace nrt {

	std::vector<std::vector<double>> NeuralNetwork::getWeights() const
	{
		std::vector<std::vector<double>> netWeights;

		for (unsigned i = 0; i < m_layers.size(); ++i) {
			for (unsigned n = 0; n < m_layers[i].size(); ++n) {
				std::vector<Connection> nWeights = m_layers[i][n].getWeights();
				std::vector<double> nOutputWeights;

				for (unsigned j = 0; j < nWeights.size(); ++j) {
					nOutputWeights.push_back(nWeights[j].weight);
				}
				netWeights.push_back(nOutputWeights);
			}
		}
		return netWeights;
	}

	inline double bval(const std::vector<Layer> l, unsigned i) {
		return l.back()[i].getOutputVal();
	}
	void NeuralNetwork::getResults(std::vector<double> &resultVals) const
	{
		// Softmax with overflow prevention
		double max = -std::numeric_limits<double>::infinity();
		for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
			if (bval(m_layers, n) > max)
				max = bval(m_layers, n);

		double Z = 0.0;
		for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
			Z += exp(bval(m_layers, n) - max);
		
		for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
			resultVals[n] = exp(bval(m_layers, n) - max) / Z;
		}
	}

	void NeuralNetwork::backProp(const std::vector<double> &targetVals)
	{
		Layer &outputLayer = m_layers.back();

		// Calculate overall NeuralNetwork error (RMS of output neuron errors)
		m_error = 0.0;

		for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
			double delta = targetVals[n] - outputLayer[n].getOutputVal();
			m_error += delta * delta;
		}
		m_error /= outputLayer.size() - 1; // Squared average error
		m_error = sqrt(m_error); // RMS (Root mean square)

								 // Recent average measurement
		m_recentAverageError =
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);

		// Calculate output layer gradients
		for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
			outputLayer[n].calcOutputGradients(targetVals[n]);
		}

		// Calculate gradients on hidden layers
		for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
			Layer &hiddenLayer = m_layers[layerNum];
			Layer &nextLayer = m_layers[layerNum + 1];

			for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
				hiddenLayer[n].calcHiddenGradients(nextLayer);
			}
		}

		// For all layers from outputs to first hidden layer, update weights
		for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
			Layer &layer = m_layers[layerNum];
			Layer &prevLayer = m_layers[layerNum - 1];

			for (unsigned n = 0; n < layer.size() - 1; ++n) {
				layer[n].updateInputWeights(prevLayer);
			}
		}
	}

	void NeuralNetwork::feedForward(const std::vector<double> &inputVals)
	{
		assert(inputVals.size() == m_layers[0].size() - 1);

		// Assign the input values into the input neurons
		for (unsigned i = 0; i < inputVals.size(); ++i) {
			m_layers[0][i].setOutputVal(inputVals[i]);
		}

		// Forward propagate
		for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
			for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
				Layer &prevLayer = m_layers[layerNum - 1];
				m_layers[layerNum][n].activate(prevLayer);
			}
		}
	}

	NeuralNetwork::NeuralNetwork(const std::vector<unsigned> &topology)
	{
		unsigned numLayers = topology.size();
		for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
			m_layers.push_back(Layer());
			unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];

			// Add neurons to layer, and bias neuron
			for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
				m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			}

			// Force bias node's output value to constant
			m_layers.back().back().setOutputVal(BIAS);
		}
	}
}