## Backpropagation 反向传播算法



Here's the general algorithm for updating the weights with backpropagation:

- Set the weight steps for each layer to zero
  + The input to hidden weights $\Delta w_{ij}=0$
  + The hidden to output weights $\Delta W_j=0$
​
- For each record in the training data:
  + Make a forward pass through the network, calculating the output $\hat y$
  + Calculate the error gradient in the output unit, $\delta^o=(y-\hat y)f'(z)$ , where $z=\sum_jW_ja_j$ , the input to the output unit.
  + Propagate the errors to the hidden layer $\delta^h_j=\delta^oW_jf'(h_j)$
  + Update the weight steps:
    * $\Delta W_j = \Delta W_j + \delta^oa_j$
    * $\Delta w_{ij} = \Delta w_{ij} + \delta^h_ja_i$
- Update the weights, where η is the learning rate and m is the number of records:
  + $W_j = W_j + \eta \Delta W_j /m$
  + $w_{ij} = w_{ij} + \eta \Delta w_{ij} /m$
- Repeat for e epochs.

vanishing gradient problem

- [Yes you should understand backprop – Medium](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
- [CS231n Winter 2016 Lecture 4 Backpropagation, Neural Networks 1-Q_UWHTY_TEQ.mp4 - YouTube](https://www.youtube.com/watch?v=59Hbtz7XgjM)
