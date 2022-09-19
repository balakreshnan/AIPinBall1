import tensorflow as tf; 
print(tf.reduce_sum(tf.random.normal([1000, 1000])))

print('Hello')

import torch
x = torch.rand(5, 3)
print(x)


torch.cuda.is_available()