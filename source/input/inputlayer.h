#pragma once

#include "inputlayerconfig.h"
#include "innetworklayer.h"
#include "layer.h"

class InputNode
{
};

class InputLayer : public Layer<InputNode, double, double, double>, public INNetworkLayer
{
private:
	int width;
	int height;
	int depth;
	int nodeCount = 0;

public:
	InputLayer(InputLayerConfig* config, INNetworkLayer* previousLayer);
	virtual void Forward(double* input, int inputSize);
	virtual void Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	virtual void Backward(double* input, int inputSize, double learnRate);
	virtual void Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
	virtual void Dispose();
	virtual double* GetForwardHostMem();
	virtual double* GetBackwardHostMem();
	virtual double* GetForwardDeviceMem();
	virtual double* GetBackwardDeviceMem();
	virtual int GetForwardNodeCount();
	virtual int GetBackwardNodeCount();
	virtual int GetWidth();
	virtual int GetHeight();
	virtual int GetDepth();
	virtual LayerType GetLayerType();
	void DebugPrint();
};
