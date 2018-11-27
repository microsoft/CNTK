# Flapping Bird using Keras and Reinforcement Learning

In [CNTK 203 tutorial](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_203_Reinforcement_Learning_Basics.ipynb), 
we have introduced the basic concepts of reinforcement
learning. In this example, we show an easy way to train a popular game called
FlappyBird using Deep Q Network (DQN). This tutorial draws heavily on the
[original work](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)
by Ben Lau on training the FlappyBird game with Keras frontend. This tutorial
uses the CNTK backend and with very little change (commenting out a few specific
  references to TensorFlow) in the original code. 

Note: Since, we have replaced the game environment components with different components drawn 
from public data sources, we call the game Flapping Bird.

# Goals

The key objective behind this example is to show how to:
- Use CNTK backend API with Keras frontend
- Interchangeably use models trained between TensorFlow and CNTK via Keras
- Train and test (evaluate) the flapping bird game using a simple DQN implementation.

# Pre-requisite

Assuming you have installed CNTK, installed Keras and configured Keras
backend to be CNTK. The details are [here](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras).

This example takes additional dependency on the following Python packages:
- pygame (`pip install pygame`)
- scikit-learn (`conda install scikit-learn`)
- scikit-image (`conda install scikit-image`)

These packages are needed to perform image manipulation operation and interactivity
of the RL agent with the game environment.

# How to run?

From the example directory, run:

```
python FlappingBird_with_keras_DQN.py -m Run
```

Note: if you run the game first time in "Run" mode a pre-trained model is
locally downloaded. Note, even though this model was trained with TensorFlow,
Keras takes care of saving and loading in a portable format. This allows for 
model evaluation with CNTK.

If you want to train the network from beginning, delete the locally cached
`FlappingBird_model.h5` (if you have a locally cached file) and run. After 
training, the trained model can be evaluated with CNTK or TensorFlow.

```
python FlappingBird_with_keras_DQN.py -m Train
```

# Brief recap

The code has 4 steps:

1. Receive the image of the game screen as pixel array
2. Process the image
3. Use a Deep Convolutional Neural Network (CNN) to predict the best action
(flap up or down)
4. Train the network (millions of times) to maximize flying time

Details can be found in Ben Lau's
[original work](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)

# Acknowledgements
- Ben Lau: Developer of Keras RL example for contributing the [code](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html) with TensorFlow backend.
- Bird sprite from [Open Game Art](https://opengameart.org/content/free-game-asset-grumpy-flappy-bird-sprite-sheets).
- Shreyaan Pathak: Seventh Grade student at Northstar Middle School, Kirkland, WA for creating and processing new game sprites.
