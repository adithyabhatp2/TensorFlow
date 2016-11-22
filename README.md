# TensorFlow

## Logging
https://www.tensorflow.org/versions/r0.11/tutorials/monitors/index.html
https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html
Finally, to write this summary data to disk, pass the summary protobuf to a tf.train.SummaryWriter.
The SummaryWriter takes a logdir in its constructor - this logdir is quite important, it's the directory where all of the events will be written out. Also, the SummaryWriter can optionally take a Graph in its constructor. If it receives a Graph object, then TensorBoard will visualize your graph along with tensor shape information. This will give you a much better sense of what flows through the graph: see Tensor shape information.
