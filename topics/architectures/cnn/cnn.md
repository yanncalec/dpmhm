# Supervised learning and CNN

Supervised learning is a subfield of machine learning where an algorithm learns to map input data to a corresponding output variable based on a set of labeled examples. In supervised learning, the algorithm is trained on a labeled dataset, which means the input data is associated with a known output or target variable. The algorithm tries to learn the relationship between the input data and the output variable, and once the model is trained, it can be used to make predictions on new, unseen data.

Supervised learning has numerous applications in condition monitoring. For instance, in predictive maintenance, the goal is to predict when a machine will fail based on sensor data collected from the machine. Supervised learning algorithms can be used to learn patterns in the sensor data that are indicative of impending machine failure. The labeled dataset used to train the algorithm would include sensor data collected from machines that have failed as well as from machines that have not yet failed. The algorithm would learn to recognize the patterns associated with machine failure and use that knowledge to predict when a machine is likely to fail in the future.

Another example of the application of supervised learning in condition monitoring is in fault diagnosis. In this case, the goal is to diagnose faults in a machine based on sensor data. Again, supervised learning algorithms can be used to learn the patterns in the sensor data associated with different types of faults. The labeled dataset used to train the algorithm would include sensor data collected from machines with known faults as well as from machines without any faults. The algorithm would learn to recognize the patterns associated with different types of faults and use that knowledge to diagnose faults in new machines.

## Resnet

ResNet (Residual Network) is a deep neural network architecture that was introduced in 2015 by Microsoft researchers Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. The architecture won the ImageNet classification challenge in 2015, and has since been widely adopted in many computer vision tasks.

ResNet uses a novel approach to overcome the problem of vanishing gradients, which occurs in very deep neural networks. The architecture introduces residual connections, which allow the network to learn the residual mapping instead of the original mapping. By doing so, the network can be deeper without negatively impacting performance.

The basic building block of ResNet is the residual block, which consists of two convolutional layers and a shortcut connection. The shortcut connection allows the output of one layer to be added to the output of another layer. This helps to ensure that the gradients can flow back through the network and avoid the problem of vanishing gradients.

ResNet comes in several different variants, with the number of layers varying between 18 and 152 layers. The most commonly used variant is ResNet-50, which has 50 layers. The architecture has been shown to achieve state-of-the-art performance on a range of image classification, object detection, and segmentation tasks.

In summary, ResNet is a deep neural network architecture that uses residual connections to allow the network to be deeper without negatively impacting performance. It has been widely adopted in many computer vision tasks and has achieved state-of-the-art performance on a range of benchmarks.

<!-- ### VGGish networks
TODO:

- explain VGGish NN and highlight some results of transfer learning on CWRU and other datasets.
- implementation details of VGGish for signal.
 -->