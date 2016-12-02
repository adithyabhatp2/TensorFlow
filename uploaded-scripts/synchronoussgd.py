import tensorflow as tf
import os
import time

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
        w = tf.Variable(tf.random_uniform([num_features, 1], minval=-10, maxval=10, name="random_init_vals", dtype=tf.float32), name="w_model")

    gradients = []
    for i in range(0, 5):
        with tf.device("/job:worker/task:%d" % i):
            filename_queue = createFileNameQueues(i)

            reader = tf.TFRecordReader()

            # Include a read operator with the filename queue to use. The output is a string
            # Tensor called serialized_example
            _, serialized_example = reader.read(filename_queue)

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
            print ""
            print "index : ", index
            print ""
            print "value : ", value
            print ""

            # for debug
            ind_vals = index.values
            ind_inds = index.indices
            val_vals = value.values

            val_vals_2d = tf.expand_dims(val_vals,1)
            w_sparse = tf.gather(w, index.values, name="w_sparse")
            w_sparse_1d = tf.reshape(w_sparse, [-1], name="w_sparse_1d")
            w_transp = tf.transpose(w_sparse)
            w_transp_1d = tf.reshape(w_transp, [-1])

            label = tf.to_float(label, name="label")
            wtranspx = tf.matmul(w_transp, val_vals_2d, name="wTransX")
            ywtx = tf.mul(label, wtranspx, name="ywtx")
            local_sigmoid = tf.sigmoid(ywtx, name="sigmoid")
            local_loss = tf.log(local_sigmoid, name="loss_intermediate")  # tensor?

            local_gradient = tf.mul(tf.mul(tf.sub(local_sigmoid, 1, name="sig_1"), val_vals, name="x_sig_1"), label, name="local_gradient_yx_sig_1")

            # dense_feature_temp = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
            #                                         [num_features, ],
            #                                         tf.sparse_tensor_to_dense(value), name="dense_feature_temp")
            # dense_feature = tf.reshape(dense_feature_temp, [num_features, 1], name="dense_feature")


            # dense_feature, label = processInputFromFile(serialized_example)

            # wtranspx = tf.matmul(tf.transpose(w), dense_feature, name="wTransX")
            # ywtx = tf.mul(label, wtranspx, name="ywtx")
            # local_sigmoid = tf.sigmoid(ywtx, name="sigmoid")
            # local_loss = tf.log(local_sigmoid, name="loss_intermediate")  # tensor?
            # ones = tf.ones([num_features, 1])
            # local_gradient = tf.mul(tf.mul(tf.sub(local_sigmoid, ones, name="sig_1"), dense_feature, name="x_sig_1"), label, name="local_gradient_yx_sig_1")
            # gradients.append(tf.mul(local_gradient, eta))

    # with tf.device("/job:worker/task:0"):
    #     aggregator = tf.add_n(gradients, name="aggr_grad")
    #     assign_op = w.assign_sub(aggregator)


    # as usual we create a session.
    with tf.Session("grpc://vm-14-1:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
    # with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        # this is new command and is used to initialize the queue based readers.
        # Effectively, it spins up separate threads to read from the files
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(0, 10):
            print i

            # output = sess.run([dense_feature, label, local_gradient])
            # print "dense_feature shape : ", output[0].shape
            # print "label : ", output[1]
            # print "local_gradient shape : ", output[2].shape

            # output = sess.run([local_gradient, assign_op])
            # print "local_gradient shape : ", output[0].shape
            # print "sum(Local gradient) : ", sum(output[0])

            # output = sess.run(local_loss)
            # print "Local Loss (Error) : ", output[0][0]

            # start = time.time()
            # sess.run(assign_op)
            # print time.time()-start

            output = sess.run([w_sparse, ind_vals, val_vals, w_sparse_1d, w_transp,w_transp_1d, val_vals_2d,wtranspx,label,ywtx, local_sigmoid, local_gradient])
            print "w_sparse", output[0].shape
            print "ind_vals", output[1].shape
            print "val_vals", output[2].shape
            print "w_sparse_1d", output[3].shape
            print "w_transp", output[4].shape
            print "w_transp_1d", output[5].shape
            print "val_vals_2d", output[6].shape
            print "w_transp_x", output[7].shape
            print "label", output[8].shape
            print "ywtx", output[9].shape
            print "local_sigmoid", output[10].shape
            print "local_gradient", output[11].shape




        coord.request_stop()
        coord.join(threads)

        tf.train.SummaryWriter("%s/sgd_sync" % (os.environ.get("TF_LOG_DIR")), sess.graph)
        sess.close()


