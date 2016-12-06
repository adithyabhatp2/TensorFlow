import tensorflow as tf
import os
import time, math

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

num_features = 33762578
eta = 0.1
neg_eta = -0.01
resultsFilePath = "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/results/async/accuracy.txt"
lastLogged=0

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

if not os.path.exists(os.path.dirname(resultsFilePath)):
    os.makedirs(os.path.dirname(resultsFilePath))

with open(resultsFilePath, "w") as f:
    f.write("Iteration\tAccuracy\n")


g = tf.Graph()

with g.as_default():

    tf.set_random_seed(1024)
    # creating a model variable on task 0. This is a process running on node vm-14-1
    with tf.device("/job:worker/task:0"):
        #w = tf.Variable(tf.random_uniform([num_features, 1], name="random_init_vals"), name="w_model")
        w = tf.Variable(tf.random_uniform([num_features, 1], minval=-10, maxval=10, name="random_init_vals"), name="w_model")
        count = tf.Variable(tf.zeros([1, 1], dtype=tf.int64), name="count")

    # Test start
    with tf.device("/job:worker/task:0"):
        test_filename_queue = tf.train.string_input_producer([
            "/home/ubuntu/gitRepository/TensorFlow/uploaded-scripts/data/criteo-tfr/tfrecords22",
        ], num_epochs=None, name="test_filename_queue")
        test_reader = tf.TFRecordReader()

        _2, test_serialized_example = test_reader.read(test_filename_queue)

        test_features = tf.parse_single_example(test_serialized_example,
                                                features={
                                                    'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                                    'index': tf.VarLenFeature(dtype=tf.int64),
                                                    'value': tf.VarLenFeature(dtype=tf.float32),
                                                })

        test_label = test_features['label']
        test_index = test_features['index']
        test_value = test_features['value']

        test_label = tf.to_float(test_label, name="test_label")

        test_ind_vals = test_index.values
        test_ind_inds = test_index.indices
        test_val_vals = test_value.values

        test_val_vals_2d = tf.expand_dims(test_val_vals, 1)
        test_w_sparse = tf.gather(w, test_index.values, name="test_w_sparse")

        test_w_transp = tf.transpose(test_w_sparse)

        test_wtranspx = tf.matmul(test_w_transp, test_val_vals_2d, name="test_wTransX")
        test_pred = tf.sign(test_wtranspx)
        test_correctness = tf.equal(test_label, test_pred, name="test_correctness")
    # Test End

    # Train start
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        filename_queue = createFileNameQueues(FLAGS.task_index)

        reader = tf.TFRecordReader()

        # Include a read operator with the filename queue to use. The output is a string
        # Tensor called serialized_example
        _, serialized_example = reader.read(filename_queue)

        # dense_feature, label = processInputFromFile(serialized_example)
        # wtranspx = tf.matmul(tf.transpose(w), dense_feature, name="wTransX")
        # ywtx = tf.mul(label, wtranspx, name="ywtx")
        # local_sigmoid = tf.sigmoid(ywtx, name="sigmoid")
        # local_loss = tf.log(local_sigmoid, name="loss_intermediate")
        # ones = tf.ones([num_features, 1])
        # local_gradient = tf.mul(tf.mul(tf.sub(local_sigmoid, ones, name="sig_1"), dense_feature, name="x_sig_1"), label, name="local_gradient_yx_sig_1")

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                               'index': tf.VarLenFeature(dtype=tf.int64),
                                               'value': tf.VarLenFeature(dtype=tf.float32),
                                           })

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

        label = tf.to_float(label, name="label")

        # for debug
        ind_vals = index.values
        ind_inds = index.indices
        val_vals = value.values

        val_vals_2d = tf.expand_dims(val_vals, 1)
        w_sparse = tf.gather(w, index.values, name="w_sparse")
        # w_sparse_1d = tf.reshape(w_sparse, [-1], name="w_sparse_1d")

        w_transp = tf.transpose(w_sparse)
        # w_transp_1d = tf.reshape(w_transp, [-1])

        wtranspx = tf.matmul(w_transp, val_vals_2d, name="wTransX")
        ywtx = tf.mul(label, wtranspx, name="ywtx")
        local_sigmoid = tf.sigmoid(ywtx, name="sigmoid")
        # local_loss = tf.log(local_sigmoid, name="loss_intermediate")  # tensor?

        local_gradient = tf.mul(tf.mul(tf.sub(local_sigmoid, 1, name="sig_1"), val_vals_2d, name="x_sig_1"), label, name="local_gradient_yx_sig_1")
        local_gradient2 = tf.mul(local_gradient, neg_eta, name="neg_eta_mult")

        local_gradient3 = tf.reshape(local_gradient2, [-1])
        # sparse_index = tf.reshape(index.values, [tf.size(index.values), num_features])
        # zeros = tf.zeros([tf.size(index.values)], dtype=tf.int64)
        # sparse_index = tf.pack([index.values, zeros], axis=1)
        #
        # lg_sparse = tf.SparseTensor(indices=sparse_index, values=local_gradient3, shape=[num_features, 1])

    with tf.device("/job:worker/task:0"):
        # w = tf.sparse_add(w, lg_sparse)
        # w = tf.scatter_sub(w, index.values, local_gradient2)
        w = tf.scatter_add(w, index.values, local_gradient2)
        count = count.assign_add(tf.ones([1, 1], dtype=tf.int64))
    # Train end

    # as usual we create a session.
    with tf.Session("grpc://vm-14-%d:2222" % (FLAGS.task_index + 1), config=tf.ConfigProto(log_device_placement=True)) as sess:
        if FLAGS.task_index == 0:
            sess.run(tf.initialize_all_variables())

        coord = tf.train.Coordinator()
        # this is new command and is used to initialize the queue based readers.
        # Effectively, it spins up separate threads to read from the files
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in xrange(0, 4000):
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
            output = sess.run([w, count])
            count_int = output[1][0][0]
            print "count :", count_int

            print time.time()-start

            test_freq = 100
            if FLAGS.task_index == 0:
                if count_int - lastLogged > test_freq or i==0:
                    num_tests = 10000
                    num_correct = 0
                    num_wrong = 0
                    test_start = time.time()
                    for test_num in xrange(0,num_tests):
                        if test_num % 1000 == 0:
                            print "test", test_num
                        output = sess.run(test_correctness)
                        if output[0][0] == True:
                            num_correct+=1
                        elif output[0][0] == False:
                            num_wrong += 1

                    # print "num_correct", num_correct
                    # print "num_wrong", num_wrong
                    accuracy = float(num_correct)/float(num_tests)
                    print "Iteration {}\tAccuracy {:.4f}".format(count_int, accuracy*100)
                    # print "Time for %d tests" % num_tests, time.time() - test_start
                    with open(resultsFilePath, "a") as f:
                        f.write("{}\t{:.4f}\n".format(count_int, accuracy*100))
                    lastLogged = (count_int/10)*10
                    print "last logged", lastLogged


        coord.request_stop()
        coord.join(threads)

        tf.train.SummaryWriter("%s/final_sgd_async" % (os.environ.get("TF_LOG_DIR")), sess.graph)


