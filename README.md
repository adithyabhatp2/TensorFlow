# TensorFlow

## Changing dims
val_vals_2d = tf.expand_dims(val_vals,1)
w_sparse_1d = tf.reshape(w_sparse, [-1], name="w_sparse_1d")

## Reading from files
https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html#AUTOGENERATED-preloaded-data
http://stackoverflow.com/questions/38678371/tensorflow-enqueue-operation-was-cancelled


## Distributed Setup
https://www.tensorflow.org/versions/r0.11/how_tos/distributed/index.html

## Logging
https://www.tensorflow.org/versions/r0.11/tutorials/monitors/index.html
https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html
Finally, to write this summary data to disk, pass the summary protobuf to a tf.train.SummaryWriter.
The SummaryWriter takes a logdir in its constructor - this logdir is quite important, it's the directory where all of the events will be written out. Also, the SummaryWriter can optionally take a Graph in its constructor. If it receives a Graph object, then TensorBoard will visualize your graph along with tensor shape information. This will give you a much better sense of what flows through the graph: see Tensor shape information.
