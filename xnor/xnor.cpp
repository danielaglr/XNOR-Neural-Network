#include "xnor.h"

int XNOR_NN::learn() {
  for (int i = 0; i < numInputs; i++) {
    for (int j = 0; j < numHiddenNodes; j++) {
      hiddenWeights[i][j] = initWeights();
    }
  }

  for (int i = 0; i < numHiddenNodes; i++) {
    for (int j = 0; j < numOutputs; j++) {
      outputWeights[i][j] = initWeights();
    }
  }

  for (int i = 0; i < numOutputs; i++) {
    outputLayerBias[i] = initWeights();
  }

  int trainingSetOrder[] = {0, 1, 2, 3};

  int numEpochs = 10000;

  for (int epoch = 0; epoch < numEpochs; epoch++) {
    std::cout << "Epoch (" << epoch << "/" << numEpochs << ") starting..." << std::endl;

    shuffle(trainingSetOrder, numTrainingSets);

    for (int x = 0; x < numTrainingSets; x++) {
      int i = trainingSetOrder[x];

      for (int j = 0; j < numHiddenNodes; j++) {
        double activation = hiddenLayerBias[j];

        for (int k = 0; k < numInputs; k++) {
          activation += trainingInputs[i][k] * hiddenWeights[k][j];
        }

        hiddenLayer[j] = sigmoid(activation);
      }

      for (int j = 0; j < numOutputs; j++) {
        double activation = outputLayerBias[j];

        for (int k = 0; k < numHiddenNodes; k++) {
          activation += hiddenLayer[k] * outputWeights[k][j];
        }

        outputLayer[j] = sigmoid(activation);
      }

      std::cout << "NN Inputs: " << trainingInputs[i][0] << ", " << trainingInputs[i][1] 
      << " Output: " << outputLayer[0]
      << " |Expected: " << trainingOutputs[i][0] << "|" 
      << std::endl;

      double deltaOuput[numOutputs];

      for (int j = 0; j < numOutputs; j++) {
        double error = (trainingOutputs[i][j] - outputLayer[j]);
        
        deltaOuput[j] = error * derivSigmoid(outputLayer[j]);
      }

      double deltaHidden[numHiddenNodes];

      for (int j = 0; j < numHiddenNodes; j++) {
        double error = 0.0f;

        for (int k = 0; k < numOutputs; k++) {
          error += deltaOuput[k] * outputWeights[j][k]; 
        }

        deltaHidden[j] = error * derivSigmoid(hiddenLayer[j]);
      }

      for (int j = 0; j < numOutputs; j++) {
        outputLayerBias[j] += deltaOuput[j] * learningRate;

        for (int k = 0; k < numHiddenNodes; k++) {
          outputWeights[k][j] += hiddenLayer[k] * deltaOuput[j] * learningRate;
        }
      }

      for (int j = 0; j < numHiddenNodes; j++) {
        hiddenLayerBias[j] += deltaHidden[j] * learningRate;

        for (int k = 0; k < numInputs; k++) {
          hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * learningRate;
        }
      }
    }

    std::cout << std::endl;
  }

  std::cout << "Final Hidden Weights:" << std::endl;
  std::cout << "[ ";
  for (int i = 0; i < numInputs; i++) {
    std::cout << "[ ";
    for (int j = 0; j < numHiddenNodes; j++) {
      std::cout << hiddenWeights[i][j] << " ";
    }
    std::cout << " ] ";
  }
  std::cout << "]" << std::endl;
  
  std::cout << "Final Hidden Biases:" << std::endl;
  std::cout << "[ ";
  for (int j = 0; j < numHiddenNodes; j++) {
    std::cout << hiddenLayerBias[j] << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << "Final Output Weights:" << std::endl;
  std::cout << "[ ";
  for (int i = 0; i < numHiddenNodes; i++) {
    for (int j = 0; j < numOutputs; j++) {
      std::cout << outputWeights[i][j] << " ";
    }
  }
  std::cout << "]" << std::endl;

  std::cout << "Final Output Biases:" << std::endl;
  std::cout << "[ ";
  for (int j = 0; j < numOutputs; j++) {
    std::cout << outputLayerBias[j] << " ";
  }
  std::cout << "]" << std::endl;

  return 0;
}

int main() {
  XNOR_NN nn;
  nn.learn();
  return 0;
}