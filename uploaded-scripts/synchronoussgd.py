import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.DEBUG)

num_features = 33762578
eta = 0.01

g = tf.Graph()

with g.as_default():

    tf.set_random_seed(1024)
    # creating a model variable on task 0. This is a process running on node vm-14-1
    with tf.device("/job:worker/task:0"):
        #w = tf.Variable(tf.ones([10, 1]), name="model")
        w = tf.Variable(tf.random_uniform([num_features, 1], minval=-100, maxval=100), name="model")


    # creating 5 reader operators to be pilaced on different operators
    # here, they emit predefined tensors. however, they can be defined as reader
    # operators as done in "exampleReadCriteoData.py"
    gradients = []
    for i in range(0, 5):
        with tf.device("/job:worker/task:%d" % i):    
            # We first define a filename queue comprising 5 files.
            filename_queue = tf.train.string_input_producer([
                "./data/criteo-tfr-tiny/tfrecords0"+str(i),
            ], num_epochs=None)
            
            
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            
            features = tf.parse_single_example(serialized_example,
                                               features={
                                                'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                                'index' : tf.VarLenFeature(dtype=tf.int64),
                                                'value' : tf.VarLenFeature(dtype=tf.float32),
                                               }
                                              )
            label = features['label']
            index = features['index']
            value = features['value']
            
            # since we parsed a VarLenFeatures, they are returned as SparseTensors.
            # To run operations on then, we first convert them to dense Tensors as below.
            # dense_feat is x
            dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
                    [num_features,],
                    #tf.constant([33762578, 1], dtype=tf.int64),
                    tf.sparse_tensor_to_dense(value))
            
            
            
            
            # reader = tf.ones([num_features, 1], name="operator_%d" % i)
            # not the gradient compuation here is a random operation. You need
            # to use the right way (as described in assignment 3 desc).
            # we use this specific example to show that gradient computation
            # requires use of the model
            # local_gradient = tf.mul(reader, tf.matmul(tf.transpose(w), reader))
            wtranspx = tf.matmul(tf.transpose(w), dense_feature)
            ywtx = tf.mul(label, wtranspx)
            local_sigmoid = tf.sigmoid(ywtx)
            local_loss = -1 * tf.log(local_sigmoid) #tensor?
            ones = tf.ones([num_features, 1])
            local_gradient = tf.mul(tf.mul(tf.subtract(local_sigmoid,ones),dense_feature),label)
            gradients.append(tf.mul(local_gradient, tf.constant(eta, shape=[num_features, 1]))
            
            
    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        aggregator = tf.add_n(gradients)
        #
        assign_op = w.assign_add(aggregator)


    with tf.Session("grpc://vm-14-1:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(0, 10):
            sess.run(assign_op)
            print w.eval()
        tf.train.SummaryWriter("%s/sgd_sync" % (os.environ.get("TF_LOG_DIR")), sess.graph)
        sess.close()

