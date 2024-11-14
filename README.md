# HandWrittenDetector
HandWrittenDetector is a neural network model that uses the TensorFlow library in Python to classify handwritten digits from the MNIST dataset. This project demonstrates building, training, and compiling a neural network for handwritten image detection.

### Model Overview
This model takes grayscale images of digits, each with dimensions of 28x28 pixels, and processes them through a neural network consisting of:

**Input Layer**: Flattened to a 1-dimensional layer with 784 neurons.

**Hidden Layers**: Two dense (fully connected) hidden layers, each with 16 neurons.

**Output Layer**: A softmax layer with 10 neurons, one for each digit (0-9).


**Key Features**

**Input Shape:** The model flattens the 28x28 image, resulting in a vector with 784 neurons.

**Hidden Layers**: Each hidden layer has 16 neurons with a sigmoid activation function, constraining the activation output between 0 and 1.

**Output:** The model's final layer outputs probabilities for each digit class using the softmax function.

**Optimizer:** Stochastic Gradient Descent (SGD) for adjusting weights via backpropagation.

**Training:** The model is trained for 5 epochs (passes over the training data).

### Note
>You must have the latest version of Python, TensorFlow library, and mnist dataset  in your system to run the code or you can run it online in Jupiter Notebook or Google Collab.
