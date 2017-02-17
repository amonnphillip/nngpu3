#include "cuda_runtime.h"
#include "nnetwork.h"
#include "inputlayer.h"
#include "inputlayerconfig.h"
#include "fullyconnectedlayer.h"
#include "relulayer.h"
#include "poollayer.h"
#include "convlayer.h"
#include "outputlayer.h"
#include <iostream>

void DebugPrintFullyConnectedLayer(FullyConnectedLayer* fullyConnectedLayer)
{
	int nodeCount = fullyConnectedLayer->GetForwardNodeCount();
	int weightCount = fullyConnectedLayer->GetWeightCount();

	std::cout << "fully connected layer:\r\n";

	std::cout << "weights:\r\n";
	for (int index = 0; index < nodeCount; index++)
	{
		double* weight = fullyConnectedLayer->GetWeightsForNode(index);

		for (int weightIndex = 0; weightIndex < weightCount; weightIndex++)
		{
			if (weightIndex + 1 != weightCount)
			{
				std::cout << *weight << " ";
			}
			else
			{
				std::cout << *weight << " : ";
			}
			weight++;
		}
	}

	std::cout << "\r\n";
	std::cout << "bias:\r\n";
	FullyConnectedNode* node = fullyConnectedLayer->GetNodeMem();
	for (int index = 0; index < nodeCount; index++)
	{
		if (index + 1 != nodeCount)
		{
			std::cout << node->bias << " ";
		}
		else
		{
			std::cout << node->bias << " ";
		}
		node++;
	}

	std::cout << "\r\n";

	std::cout << "forward:\r\n";
	double* output = fullyConnectedLayer->GetForwardHostMem();
	for (int index = 0; index < nodeCount; index++)
	{
		std::cout << *output << " ";
		output++;
	}

	std::cout << "\r\n\r\n";
}

int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	// Create the (very small) network
	NNetwork* nn = new NNetwork();
	nn->Add<InputLayer, InputLayerConfig>(new InputLayerConfig(8, 8, 1));
	//nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(8, 8, 1));
	nn->Add<ConvLayer, ConvLayerConfig>(new ConvLayerConfig(3, 3, 1, 4, 1, 1));
	nn->Add<ReluLayer>();
	nn->Add<PoolLayer, PoolLayerConfig>(new PoolLayerConfig(2, 2));
	//nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(2));
	nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(2));
	nn->Add<OutputLayer, OutputLayerConfig>(new OutputLayerConfig(2));

	// Train the network
	int iterationCount = 0;
	int interationMax = 3000;
	while (iterationCount < interationMax)
	{
		const int inputCount = 64;
		const int expectedCount = 2;
		double* input;
		double* expected;

		if (iterationCount & 1)
		{
			double inputAlt[] = {
				1, 1,  0, 0,  0, 0, 0, 0,
				1, 1,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
			};
			input = inputAlt;

			double expectedAlt[] = { 1, 0 };
			expected = expectedAlt;
		}
		else
		{
			double inputAlt[] = {
				0, 0,  1, 1,  0, 0, 0, 0,
				0, 0,  1, 1,  0, 0, 0, 0,
				1, 1,  0, 0,  0, 0, 0, 0,
				1, 1,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
				0, 0,  0, 0,  0, 0, 0, 0,
			};
			input = inputAlt;

			double expectedAlt[] = { 0, 1 };
			expected = expectedAlt;
		}


		nn->Forward(input, inputCount);
		nn->Backward(expected, expectedCount, 0.01);


		std::cout << "iteration: " << iterationCount << "\r\n";
		for (int layerIndex = 0; layerIndex < nn->GetLayerCount(); layerIndex++)
		{
			INNetworkLayer* layer = nn->GetLayer(layerIndex);
			LayerType layerType = layer->GetLayerType();
			if (layerType == LayerType::Input)
			{
				double* forward = layer->GetForwardHostMem();
				int forwardCount = layer->GetForwardNodeCount();
				std::cout << "input:\r\n";
				for (int index = 0; index < forwardCount; index++)
				{
					std::cout << forward[index] << " ";
				}
			}
			else if (layerType == LayerType::Convolution)
			{
				ConvLayer* convLayer = dynamic_cast<ConvLayer*>(layer);
				convLayer->DebugPrint();
			}
			else if (layerType == LayerType::FullyConnected)
			{
				FullyConnectedLayer* fullyConnectedLayer = dynamic_cast<FullyConnectedLayer*>(layer);
				DebugPrintFullyConnectedLayer(fullyConnectedLayer);
			}
			else if (layerType == LayerType::Output)
			{
				double* forward = layer->GetForwardHostMem();
				int forwardCount = layer->GetForwardNodeCount();
				std::cout << "output:\r\n";
				for (int index = 0; index < forwardCount; index++)
				{
					std::cout << forward[index] << " ";
				}
				std::cout << "\r\n";
				std::cout << "expected:\r\n";
				for (int index = 0; index < forwardCount; index++)
				{
					std::cout << expected[index] << " ";
				}
			}

			std::cout << "\r\n";
		}

		std::cout << "\r\n\r\n";

		iterationCount++;
	}
	
	// Dispose of the resouces we allocated and close
	nn->Dispose();
	delete nn;

	cudaDeviceReset();
}

