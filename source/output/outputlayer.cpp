#include <cassert>
#include <iostream>
#include "outputlayer.h"
#include "layerexception.h"
#include "cuda_runtime.h"
#include "layer.h"


OutputLayer::OutputLayer(OutputLayerConfig* config, INNetworkLayer* previousLayer)
{
	nodeCount = config->GetWidth() * config->GetHeight() * config->GetDepth();
	Layer::Initialize(
		LayerType::Output,
		nodeCount,
		nodeCount,
		0,
		true);
}

void OutputLayer::Dispose()
{
	Layer::Dispose();
}

void OutputLayer::Forward(double* input, int inputSize)
{
	throw LayerException("Forward variant not valid for OutputLayer layer");
}

void OutputLayer::Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer)
{
	assert(previousLayer->GetForwardNodeCount() == nodeCount);

	memcpy(forwardHostMem.get(), previousLayer->GetForwardHostMem(), nodeCount * sizeof(double));
}

void OutputLayer::Backward(double* input, int inputSize, double learnRate)
{
	assert(inputSize == nodeCount);

	// TODO: Optimize
	double* forward = forwardHostMem.get();
	double* backward = backwardHostMem.get();
	for (int index = 0; index < nodeCount; index++)
	{
		*backward = *input - *forward;
		forward++;
		input++;
		backward++;
	}

	if (cudaMemcpy(backwardDeviceMem, backwardHostMem.get(), nodeCount * sizeof(double), cudaMemcpyHostToDevice) != cudaError::cudaSuccess)
	{
		throw std::runtime_error("OutputLayer backward cudaMemcpy returned an error");
	}
}

void OutputLayer::Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate)
{
	throw LayerException("Backward variant not valid for OutputLayer layer");
}

double* OutputLayer::GetForwardHostMem()
{
	return forwardHostMem.get();
}

double* OutputLayer::GetBackwardHostMem()
{
	return backwardHostMem.get();
}

double* OutputLayer::GetForwardDeviceMem()
{
	return forwardDeviceMem;
}

double* OutputLayer::GetBackwardDeviceMem()
{
	return backwardDeviceMem;
}

int OutputLayer::GetForwardNodeCount()
{
	return nodeCount;
}

int OutputLayer::GetBackwardNodeCount()
{
	return nodeCount;
}

int OutputLayer::GetWidth()
{
	return nodeCount;
}

int OutputLayer::GetHeight()
{
	return 1;
}

int OutputLayer::GetDepth()
{
	return 1;
}

LayerType OutputLayer::GetLayerType()
{
	return Layer::GetLayerType();
}

void OutputLayer::DebugPrint(double* expected, int expectedCount)
{
	assert(expectedCount == GetForwardNodeCount());

	double* forward = GetForwardHostMem();
	int forwardCount = GetForwardNodeCount();
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