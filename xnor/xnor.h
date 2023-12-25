#ifndef XNOR_H
#define XNOR_H

#include <iostream>
#include <math.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

class XNOR_NN {
  const double learningRate = 0.1f;

  double hiddenLayer[numHiddenNodes];
  double outputLayer[numOutputs];

  double hiddenLayerBias[numHiddenNodes];
  double outputLayerBias[numOutputs];

  double hiddenWeights[numInputs][numHiddenNodes];
  double outputWeights[numHiddenNodes][numOutputs];

  double trainingInputs[numTrainingSets][numInputs] = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 1.0f}
  };

  double trainingOutputs[numTrainingSets][numOutputs] = {
    {1.0f},
    {0.0f},
    {0.0f},
    {1.0f}
  };

  void shuffle(int *arr, size_t n) {
    if (n > 1) {
      size_t i;

      for (i = 0; i < n - 1; i++) {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        int val = arr[j];
        arr[j] = arr[i];
        arr[i] = val;
      }
    }
  }

  double initWeights() { return ((double)rand()) / ((double)RAND_MAX); };
  double sigmoid(double x) { return 1 / (1 + exp(-x)); };
  double derivSigmoid(double x) { return x * (1 - x); };

public:
  int learn();
};

#endif