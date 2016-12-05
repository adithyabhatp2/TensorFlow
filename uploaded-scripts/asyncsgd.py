import tensorflow as tf
import os
import time

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

num_features = 33762578
eta = 0.01

def processInputFromFile(serialized_example):
    # The string tensors is essentially a Protobuf serialized string. With the
    # following fields: label, index, value. We provide the protobuf fields we are
    # interested in to parse the data. Note, feature here is a dict of tensors
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                           'index': tf.VarLenFeature(dtype=tf.int64),
                                           'value': tf.VarLenFeature(dtype=tf.float32),
                                       }
                                       )
    label = features['label']
    index = features['index']
    value = features['value']

    # These print statements are there for you see the type of the following
    # variables
    print "label : ", label
    print "index : ", index
    print "value : ", value
    print ""

    # since we parsed a VarLenFeatures, they are returned as SparseTensors.
    # To run operations on then, we first convert them to dense Tensors as below.
    dense_feature_temp = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
                                            [num_features, ],
                                            tf.sparse_tensor_to_dense(value), name="dense_feature_temp")
    dense_feature = tf.reshape(dense_feature_temp, [num_features, 1], name="dense_feature")
    label = tf.to_float(label, name="label")
    return dense_feature, label


def createFileNameQueues(machine_id):
    start = machine_id*5
    if machine_id < 2:
        filename_queue = tf.train.string_input_producer([
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords0" + str(start+0),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords0" + str(start+1),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords0" + str(start+2),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords0" + str(start+3),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords0" + str(start+4),
        ], num_epochs=None, name="filename_queue_"+str(machine_id))
    elif machine_id < 4:
        filename_queue = tf.train.string_input_producer([
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords" + str(start + 0),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords" + str(start + 1),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords" + str(start + 2),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords" + str(start + 3),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords" + str(start + 4),
        ], num_epochs=None, name="filename_queue_" + str(machine_id))
    else:
        filename_queue = tf.train.string_input_producer([
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords" + str(start + 0),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr-tiny/tfrecords" + str(start + 1),
        ], num_epochs=None, name="filename_queue_" + str(machine_id))
    return filename_queue


g = tf.Graph()

with g.as_default():

    tf.set_random_seed(1024)
    # creating a model variable on task 0. This is a process running on node vm-14-1
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.random_uniform([num_features, 1], minval=-10, maxval=10, name="random_init_vals"), name="w_model")

    for i in range(0, 5):
        with tf.device("/job:worker/task:%d" % i):
            filename_queue = createFileNameQueues(i)

            reader = tf.TFRecordReader()

            # Include a read operator with the filename queue to use. The output is a string
            # Tensor called serialized_example
            _, serialized_example = reader.read(filename_queue)

            dense_feature, label = processInputFromFile(serialized_example)

            wtranspx = tf.matmul(tf.transpose(w), dense_feature, name="wTransX")
            ywtx = tf.mul(label, wtranspx, name="ywtx")
            local_sigmoid = tf.sigmoid(ywtx, name="sigmoid")
            local_loss = tf.log(local_sigmoid, name="loss_intermediate")
            ones = tf.ones([num_features, 1])
            local_gradient = tf.mul(tf.mul(tf.sub(local_sigmoid, ones, name="sig_1"), dense_feature, name="x_sig_1"), label, name="local_gradient_yx_sig_1")

    with tf.device("/job:worker/task:0"):
        assign_op = w.assign_sub(local_gradient)


    # as usual we create a session.
    with tf.Session("grpc://vm-14-1:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
        if FLAGS.task_index == 0:
            sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        # this is new command and is used to initialize the queue based readers.
        # Effectively, it spins up separate threads to read from the files
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(0, 20):
            print i
            # every time we call run, a new data point is read from the files
            #output = sess.run([dense_feature, label, local_gradient])
            # print "dense_feature shape : ", output[0].shape
            # print "sum(dense_feature) : ", sum(output[0])
            # print "label : ", output[1]
            # print "local_gradient shape : ", output[2].shape

            # output = sess.run([local_gradient, assign_op])
            # print "local_gradient shape : ", output[0].shape
            # print "sum(Local gradient) : ", sum(output[0])

            # output = sess.run(local_loss)
            # print "Local Loss (Error) : ", output[0][0]
            start = time.time()
            sess.run(assign_op)
            print time.time()-start




        coord.request_stop()
        coord.join(threads)

        tf.train.SummaryWriter("%s/sgd_async" % (os.environ.get("TF_LOG_DIR")), sess.graph)


