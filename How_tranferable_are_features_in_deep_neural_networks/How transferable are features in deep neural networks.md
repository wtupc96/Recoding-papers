# How transferable are features in deep neural networks? 

- **Existing problem:**
  - Many deep neural networks trained on natural images exhibit a curious phenomenon in common: on the first layer they learn features similar to Gabor filters and color blobs. Such first-layer features appear not to be *specific* to a particular dataset or task, but *general* in that they are applicable to many datasets and tasks. Features must eventually transition from general to specific by the last layer of the network, but this transition has not been studied extensively.
- **Main idea:**
  - We define the degree of generality of a set of features learned on the task A as the extent to which the features can be used for another task B.
  - There are two ways to learn parameters:
    - Learn all the parameters in the network;
    - Learn only part B parameters.