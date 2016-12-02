import tensorflow as tf
import os
import time
import numpy

tf.logging.set_verbosity(tf.logging.DEBUG)

# number of features in the criteo dataset after one-hot encoding
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
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords0" + str(start+0),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords0" + str(start+1),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords0" + str(start+2),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords0" + str(start+3),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords0" + str(start+4),
        ], num_epochs=None, name="filename_queue_"+str(machine_id))
    elif machine_id < 4:
        filename_queue = tf.train.string_input_producer([
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords" + str(start + 0),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords" + str(start + 1),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords" + str(start + 2),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords" + str(start + 3),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords" + str(start + 4),
        ], num_epochs=None, name="filename_queue_" + str(machine_id))
    else:
        filename_queue = tf.train.string_input_producer([
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords" + str(start + 0),
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords" + str(start + 1),
        ], num_epochs=None, name="filename_queue_" + str(machine_id))
    return filename_queue


def createTestfileNameQueue():
    filename_queue = tf.train.string_input_producer([
        "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords22",
    ], num_epochs=None, name="test_filename_queue")
    return filename_queue


def processTestInput(serialized_example, num_records):

    features = tf.parse_example(serialized=serialized_example,
                                        features={
                                           'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                           'index': tf.VarLenFeature(dtype=tf.int64),
                                           'value': tf.VarLenFeature(dtype=tf.float32),
                                       }, name="test_parse")
    label = features['label']
    index = features['index']
    value = features['value']

    # These print statements are there for you see the type of the following
    # variables
    print "Test label : ", label
    print "Test index : ", index
    print "Test value : ", value
    print ""

    # since we parsed a VarLenFeatures, they are returned as SparseTensors.
    # To run operations on then, we first convert them to dense Tensors as below.
    # dense_index = tf.sparse_tensor_to_dense(index, name="dense_index")
    # print dense_index
    # dense_value = tf.sparse_tensor_to_dense(value, name="dense_value")
    # print dense_value
    # dense_feature_temp = tf.sparse_to_dense(dense_index, [num_features, 2], dense_value, name="dense_feature_temp")

    dense_feature_temp = tf.sparse_to_dense(index.indices, [num_records, num_features], value.values)
    dense_feature = tf.reshape(dense_feature_temp, [num_records, num_features], name="dense_feature")
    label = tf.to_float(label, name="label")
    return dense_feature, label


g = tf.Graph()

with g.as_default():

    tf.set_random_seed(1024)
    # creating a model variable on task 0. This is a process running on node vm-14-1
    # with tf.device("/job:worker/task:0"):
    w = tf.Variable(tf.random_uniform([num_features, 1], minval=-10, maxval=10, name="random_init_vals"), name="w_model")
    loss = tf.Variable(tf.zeros([1,1]), name="loss")

    num_records=10

    test_filename_queue = createTestfileNameQueue()
    test_reader = tf.TFRecordReader()
    # _, test_serialized_collection = test_reader.read_up_to(test_filename_queue, num_records=num_records)
    # test_x, test_y = processTestInput(test_serialized_collection,num_records)
    _, test_serialized_collection = test_reader.read(test_filename_queue)
    test_x, test_y = processInputFromFile(test_serialized_collection)
    test_wtranspx = tf.matmul(tf.transpose(w), test_x, name="test_wtransx")

    test_pred = tf.sign(test_wtranspx)
    test_correctness = tf.equal(test_y, test_pred, name="test_correctness")
    test_pos = tf.reduce_sum(tf.cast(test_correctness, tf.float32))



    gradients = []

    filename_queue = createFileNameQueues(3)

    # TFRecordReader creates an operator in the graph that reads data from queue
    reader = tf.TFRecordReader()

    # Include a read operator with the filename queue to use. The output is a string
    # Tensor called serialized_example
    _, serialized_example = reader.read(filename_queue)

    dense_feature, label = processInputFromFile(serialized_example)

    wtranspx = tf.matmul(tf.transpose(w), dense_feature, name="wTransX")
    ywtx = tf.mul(label, wtranspx, name="ywtx")
    local_sigmoid = tf.sigmoid(ywtx, name="sigmoid")
    local_loss = tf.log(local_sigmoid, name="loss_intermediate")  # tensor?
    ones = tf.ones([num_features, 1])
    local_gradient = tf.mul(tf.mul(tf.sub(local_sigmoid, ones, name="sig_1"), dense_feature, name="x_sig_1"), label, name="local_gradient_yx_sig_1")
    gradients.append(tf.mul(local_gradient, eta))

    aggregator = tf.add_n(gradients, name="aggr_grad")
    assign_op = w.assign_sub(aggregator)


    # as usual we create a session.
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        # this is new command and is used to initialize the queue based readers.
        # Effectively, it spins up separate threads to read from the files
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(0, 40):
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
            print "Time : ", time.time()-start



            if i%10 == 0:
            #     start = time.time()
            #     output = sess.run(test_y)
            #     print "Test time: ", time.time() - start
            #     print "test.y shape", output.shape
            #     unique, counts = numpy.unique(output, return_counts=True)
            #     print "Test distribution", dict(zip(unique, counts))
            #
                correct = 0
                num_tests = 100
                for test_num in xrange(0,num_tests):
                    print test_num
                    output2 = sess.run(test_pos)
                    correct += output2
                print "test.accuracy :", (correct/num_tests)








        coord.request_stop()
        coord.join(threads)

        tf.train.SummaryWriter("%s/sync_single" % (os.environ.get("TF_LOG_DIR")), sess.graph)


