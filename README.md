### Artificial Neural Network (ANN)?

Artificial neural network (ANN) is a computational model that consists of several processing elements that receive inputs and deliver outputs based on their predefined activation functions.

![ann](./ann.jpg)

#### Activation Function:


There are four types of activation function.

1. Threshold function:

    The output is set at one of two levels, depending on whether the total input is greater than or less than some threshold value.

    ![threshold](./threshold.png)

2. Sigmoid function:

    The values of logistic function range from 0 and 1 

    ![sigmoid](./sigmoid.JPG)

3. Rectifier  Function:

    ![rectifier](./rectifier.jpg)

4. Hyperbolic tangent function:

    ![tangent](./tangent.jpg)

[Types of activation functions][1]

[1]:https://www.v7labs.com/blog/neural-networks-activation-functions "activation functions"

#### What is a loss/Cost function?


‘Loss’ in Machine learning helps us understand the difference between the predicted value & the actual value. The Function used to quantify this loss during the training phase in the form of a single real number is known as “Loss Function”. These are used in those supervised learning algorithms that use optimization techniques. Notable examples of such algorithms are regression, logistic regression, etc. The terms cost function & loss function are analogous.


Loss function:  Used when we refer to the error for a single training example.

Cost function: Used to refer to an average of the loss functions over an entire training dataset.

[Cost function is no rocket science!][2]

[2]:https://www.analyticsvidhya.com/blog/2021/02/cost-function-is-no-rocket-science/

#### What is Backpropagation?

Backpropagation is the essence of neural network training. It is the method of fine-tuning the weights of a neural network based on the error rate obtained in the previous epoch (i.e., iteration). Proper tuning of the weights allows you to reduce error rates and make the model reliable by increasing its generalization.

![backpropagation](./backpropagation.jpg)

[How Does Back-Propagation in Artificial Neural Networks Work?][3]

[How the backpropagation algorithm works][4]

[3]:https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea7
[4]:http://neuralnetworksanddeeplearning.com/chap2.html

#### What is Gradiant descent?

Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. Gradient descent is simply used in machine learning to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.


"A gradient measures how much the output of a function changes if you change the inputs a little bit." — Lex Fridman (MIT)

[Gradiant Descent][5]

[5]:https://builtin.com/data-science/gradient-descent

![gradiant-descent](./gradiant-descent.jpg)

Types of Gradient Descent

There are three popular types of gradient descent that mainly differ in the amount of data they use: 


* Batch Gradient Descent

    Batch gradient descent, also called vanilla gradient descent, calculates the error for each example within the training dataset, but only after all training examples have been evaluated does the model get updated. This whole process is like a cycle and it's called a training epoch.


    Some advantages of batch gradient descent are its computational efficient, it produces a stable error gradient and a stable convergence. Some disadvantages are the stable error gradient can sometimes result in a state of convergence that isn’t the best the model can achieve. It also requires the entire training dataset be in memory and available to the algorithm.


* Stochastic gradient descent

    By contrast, stochastic gradient descent (SGD) does this for each training example within the dataset, meaning it updates the parameters for each training example one by one. Depending on the problem, this can make SGD faster than batch gradient descent. One advantage is the frequent updates allow us to have a pretty detailed rate of improvement.


    The frequent updates, however, are more computationally expensive than the batch gradient descent approach. Additionally, the frequency of those updates can result in noisy gradients, which may cause the error rate to jump around instead of slowly decreasing.


* Mini-batch gradient descent

    Mini-batch gradient descent is the go-to method since it’s a combination of the concepts of SGD and batch gradient descent. It simply splits the training dataset into small batches and performs an update for each of those batches. This creates a balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent.
