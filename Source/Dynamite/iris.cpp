#include <vector>

#include <boost/filesystem.hpp>

#include "marian-cntk.h"
#include "helper.h"

using namespace marian;
//using namespace data;
//using namespace keywords;

// Constants for Iris example
const size_t MAX_EPOCHS = 200;

// Function creating feedforward dense network graph
Expr buildIrisClassifier(Ptr<ExpressionGraph> graph,
                         std::vector<float> inputData,
                         std::vector<float> outputData = {},
                         bool train = false) {
  // The number of input data
  int N = inputData.size() / NUM_FEATURES;

  graph->clear();

  // Define the input layer
  auto x = graph->constant({N, NUM_FEATURES},
                           /*init =*/ inits::from_vector(inputData));
  x->log();

  // Define the hidden layer
  auto W1 = graph->param("W1", {NUM_FEATURES, 5}, init = inits::uniform());
  auto b1 = graph->param("b1", {1, 5}, init = inits::zeros);
  auto h = tanh(affine(x, W1, b1));
  W1->log();
  b1->log();
  h->log();

  // Define the output layer
  auto W2 = graph->param("W2", {5, NUM_LABELS}, init = inits::uniform());
  auto b2 = graph->param("b2", {1, NUM_LABELS}, init = inits::zeros);
  auto o = affine(h, W2, b2);
  W2->log();
  b2->log();
  o->log();

  if(train) {
    auto y = graph->constant({N}, /*init =*/ inits::from_vector(outputData));
    y->log();
    /* Define cross entropy cost on the output layer.
     * It can be also defined directly as:
     *   -mean(sum(logsoftmax(o) * y, axis=1), axis=0)
     * But then `y` requires to be a one-hot-vector, i.e. [0,1,0, 1,0,0, 0,0,1,
     * ...] instead of [1, 0, 2, ...].
     */
    auto cost = mean(cross_entropy(o, y), axis = 0);
    cost->log();
    return cost;
  } else {
    auto preds = logsoftmax(o);
    return preds;
  }
}

//int main()
int iris_main()
{
  // Initialize global settings
  //createLoggers();

  // Disable randomness by setting a fixed seed for random number generator
  //Config::seed = 123456;

  // Get path do data set
  std::string dataPath
      = (boost::filesystem::path(__FILE__).parent_path() / "iris.data")
            .string();

  // Read data set (all 150 examples)
  std::vector<float> trainX;
  std::vector<float> trainY;
  readIrisData(dataPath, trainX, trainY);

  // Split shuffled data into training data (120 examples) and test data (rest
  // 30 examples)
  shuffleData(trainX, trainY);
  std::vector<float> testX(trainX.end() - 30 * NUM_FEATURES, trainX.end());
  trainX.resize(120 * NUM_FEATURES);
  std::vector<float> testY(trainY.end() - 30, trainY.end());
  trainY.resize(120);

  {
    // Create network graph
    auto graph = New<ExpressionGraph>();

    // Set general options
    graph->setDevice(0);
    graph->reserveWorkspaceMB(128);

    // Choose optimizer (Sgd, Adagrad, Adam) and initial learning rate
    //auto opt = Optimizer<Adam>(0.005);
    auto opt = Optimizer<Sgd>(0.005*20); // [fseide] using default SGD to minimize differences

    for(size_t epoch = 1; epoch <= MAX_EPOCHS; ++epoch) {
      // Shuffle data in each epochs
      shuffleData(trainX, trainY);

      // Build classifier
      auto cost = buildIrisClassifier(graph, trainX, trainY, true);

      // Train classifier and update weights
      graph->backprop(cost); // must break compat here
      opt->update(graph);

      if(epoch % 10 == 0)
        std::cout << "Epoch: " << epoch << " Cost: " << cost->scalar()
                  << std::endl;
    }

    // Build classifier with test data
    auto probs = buildIrisClassifier(graph, testX);

    // Print probabilities for debugging. The `debug` function has to be called
    // prior to computations in the network.
    // debug(probs, "Classifier probabilities")

    // Run classifier
    //graph->forward();

    // Extract predictions
    std::vector<float> preds(testY.size());
    // TODO: add an overload "get" in place of CopyDataTo()
    //probs->val()->get(preds);
    probs->val()->CopyDataTo(preds);

    std::cout << "Accuracy: " << calculateAccuracy(preds, testY) << std::endl;
  }

  return 0;
}
