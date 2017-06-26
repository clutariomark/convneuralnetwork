import os
import random
import tensorflow as tf
import cv2
import numpy as np
import time
import csv
from sklearn.model_selection import KFold
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib import slim

def get_image_and_labels(data_dir, all=True):
    image_dir = os.path.join(data_dir, "Images")
    emotion_dir = os.path.join(data_dir, "Emotion")
    
    image = []
    label = []
    
    for root, dirs, files in os.walk(emotion_dir):
        file_len = len(files)
        if file_len != 0:
            for file in files:
                basefile, ext = os.path.splitext(file)
                if ext == '.txt':
                    base = basefile.rsplit('_', 1)[0]
                    basedirs = base.split('_')
                    image_num = int(basedirs[2])
                
                    emotion_file = os.path.join(emotion_dir, basedirs[0], basedirs[1], file)
                
                    f = open(emotion_file, 'r')
                    for line in f:
                        line = line.strip()
                        line = int(float(line))
                        line = line - 1

                    if all == True:
                        temp_imagedir = os.path.join(image_dir, basedirs[0], basedirs[1])
                        for root, dirs, files in os.walk(temp_imagedir):
                            for file in files:
                                bfile, ext = os.path.splitext(file)
                                if ext == '.png':
                                    image.append(os.path.join(temp_imagedir, file))
                                    label.append(line)
                    else:
                        temp_imagedir_file = os.path.join(image_dir, basedirs[0], basedirs[1], "%s.png" % base)
                        image.append(temp_imagedir_file)
                        label.append(line)
                        
    return image, label

def process_images(images):
    proc_images = []
    index = 0
    for image in images:
        image = cv2.imread(image, 0)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC)
        proc_images.append(image)
        index += 1

    np.save("processed_images.npy", proc_images)
    return proc_images

def process_labels(labels):
    proc_labels = []
    for label in labels:
        one_hot = np.zeros(CLASSES)
        one_hot[label] = 1.0
        proc_labels.append(one_hot)
    np.save("processed_labels.npy", proc_labels)
    return proc_labels

def do_kfold(proc_images, proc_labels, split=10):
    trainimages = []
    trainlabels = []
    testimages = []
    testlabels = []
    rand_idx = random.sample(range(0, len(proc_images)), len(proc_images))
    proc_images = proc_images[rand_idx]
    proc_labels = proc_labels[rand_idx]
    kf = KFold(n_splits=split)
    for train_index, test_index in kf.split(proc_images):
        x_train, x_test = proc_images[train_index], proc_images[test_index]
        y_train, y_test = proc_labels[train_index], proc_labels[test_index]
        trainimages.append(x_train)
        testimages.append(x_test)
        trainlabels.append(y_train)
        testlabels.append(y_test)

    np.save("trainimages.npy", trainimages)
    np.save("testimages.npy", testimages)
    np.save("trainlabels.npy", trainlabels)
    np.save("testlabels.npy", testlabels)
    return(trainimages, testimages, trainlabels, testlabels)
    
def conv2d(x, W, b, strides=1, padding=0, name=None):
    # Conv2D wrapper, with bias and relu activation
    if padding != 0:
        x = tf.pad(x, [[0,0],[padding,padding],[padding,padding],[0,0]])
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID', name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
def maxpool2d(x, k=2, strides=1, padding=0):
    # MaxPool2D wrapper
    if padding != 0:
        x = tf.pad(x, [[0,0],[padding,padding],[padding,padding],[0,0]])
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,strides,strides,1], padding='VALID')
    
def conv_net(x, filters, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    # Convolution 1
    conv1 = conv2d(x, filters['1'], biases['1'], strides=2, padding=3, name="CONV1")
    # Pooling 1
    pool1 = maxpool2d(conv1, k=3, strides=2, padding=1)
    # LRN
    lrn = tf.nn.lrn(pool1)
    # Convolution 2a
    conv2a = conv2d(lrn, filters['2a'], biases['2a'], strides=1, padding=0, name="CONV2A")
    # Pooling 2a
    pool2a = maxpool2d(lrn, k=3, strides=1, padding=1)
    # Convolution 2b
    conv2b = conv2d(conv2a, filters['2b'], biases['2b'], strides=1, padding=1, name="CONV2B")
    # Convolution 2c
    conv2c = conv2d(pool2a, filters['2c'], biases['2c'], strides=1, padding=0, name="CONV2C")
    # Concatenate 2
    concat2 = tf.concat(3, [conv2b, conv2c])
    # Pooling 2b
    pool2b = maxpool2d(concat2, k=3, strides=2, padding=1)
    # Convolution 3a
    conv3a = conv2d(pool2b, filters['3a'], biases['3a'], strides=1, padding=0, name="CONV3A")
    # Pooling 3a
    pool3a = maxpool2d(pool2b, k=3, strides=1, padding=1)
    
    # Convolution 3b
    conv3b = conv2d(conv3a, filters['3b'], biases['3b'], strides=1, padding=1, name="CONV3B")
    # Convolution 3c
    conv3c = conv2d(pool3a, filters['3c'], biases['3c'], strides=1, padding=0, name="CONV3C")
    # Concatenate 3
    concat3 = tf.concat(3, [conv3b, conv3c])
    # Pooling 3b
    pool3b = maxpool2d(concat3, k=3, strides=2, padding=1)
    # Fully Connected Layer
    pool_shape = pool3b.get_shape().as_list()
    fc = tf.reshape(
        pool3b, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]]
    )
    fc = tf.add(tf.matmul(fc, filters['fc']), biases['fc'])
    # Apply Dropout
    fc = tf.nn.dropout(fc, dropout)
    # Output/Prediction
    out = tf.add(tf.matmul(fc, filters['out']), biases['out'])
    
    return out

# Parameters
LEARNING_RATE = 0.001
ITERATIONS = 50
BATCH_SIZE = 32
DISPLAY_STEP = 1
CLASSES = 7
CHANNELS = 1
IMAGE_SIZE = 224
N_INPUT = IMAGE_SIZE * IMAGE_SIZE
DROPOUT = 1.0

# Store layers weight and bias
filters = {
    '1': tf.Variable(tf.truncated_normal([7,7,CHANNELS,64], stddev=0.01), name="F1"),
    '2a': tf.Variable(tf.truncated_normal([1,1,64,96], stddev=0.01), name="F2a"),
    '2b': tf.Variable(tf.truncated_normal([3,3,96,208],stddev=0.01), name="F2b"),
    '2c': tf.Variable(tf.truncated_normal([1,1,64,64],stddev=0.01), name="F2c"),
    '3a': tf.Variable(tf.truncated_normal([1,1,272,96],stddev=0.01), name="F3a"),
    '3b': tf.Variable(tf.truncated_normal([3,3,96,208],stddev=0.01), name="F3b"),
    '3c': tf.Variable(tf.truncated_normal([1,1,272,64],stddev=0.01), name="F3c"),
    'fc': tf.Variable(tf.truncated_normal([272*14*14, 11],stddev=0.01), name="FC"),
    'out': tf.Variable(tf.truncated_normal([11, CLASSES],stddev=0.01), name="Out")
}

biases = {
    '1': tf.Variable(tf.truncated_normal([64], stddev=0.01), name="B1"),
    '2a': tf.Variable(tf.truncated_normal([96], stddev=0.01), name="B2a"),
    '2b': tf.Variable(tf.truncated_normal([208],stddev=0.01), name="B2b"),
    '2c': tf.Variable(tf.truncated_normal([64],stddev=0.01), name="B2c"),
    '3a': tf.Variable(tf.truncated_normal([96],stddev=0.01), name="B3a"),
    '3b': tf.Variable(tf.truncated_normal([208],stddev=0.01), name="B3b"),
    '3c': tf.Variable(tf.truncated_normal([64],stddev=0.01), name="B3c"),
    'fc': tf.Variable(tf.truncated_normal([11],stddev=0.01), name="BFC"),
    'out': tf.Variable(tf.truncated_normal([CLASSES],stddev=0.01), name="B_Out")
}

# Get File Paths and Labels
env_dir = os.getcwd()
# data_dir = os.path.join(env_dir, "data")
data_dir = env_dir
log_dir = os.path.join(env_dir, "logdir")
images, labels = get_image_and_labels(data_dir, True)

# Load processed data
#proc_images = np.load("processed_images.npy")
#proc_labels = np.load("processed_labels.npy")

print("Saving processed data")
# trainimages, testimages, trainlabels, testlabels = do_kfold(proc_images, proc_labels, split=10)

print("Loading processed data")
trainimages = np.load("trainimages.npy")
testimages = np.load("testimages.npy")
trainlabels = np.load("trainlabels.npy")
testlabels = np.load("testlabels.npy")

# tf Graph Input
x = tf.placeholder(tf.float32, [None, N_INPUT])
y = tf.placeholder(tf.float32, [None, CLASSES])
keep_prob = tf.placeholder(tf.float32)

# Contruct Model
out = conv_net(x, filters, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
pred = tf.argmax(out, 1)
lab = tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init1 = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()

runtimes = []
# Select K Images and Labels
for k in range(10):
    start = time.clock()
    print("**************** kfold: %d ****************" % k)
    cost_periter = []
    acc_periter = []
    test_acc = []
    test_prediction = []
    test_label = []
    chkptfile = os.path.join(log_dir, "model%d.ckpt" % k)
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run([init1, init2])

        # Training Part 
        for iter in range(ITERATIONS):
            tic = time.clock()
            train_len = len(trainimages[k])
            test_len = len(testimages[k])
            tmp_acc = []
            tmp_cost = []
            for num in range(train_len/BATCH_SIZE):
                if num*BATCH_SIZE+BATCH_SIZE <= train_len:
                    end_batch = num*BATCH_SIZE+BATCH_SIZE
                else:
                    end_batch = train_len
                image_batch = trainimages[k][num*BATCH_SIZE:end_batch]
                label_batch = trainlabels[k][num*BATCH_SIZE:end_batch]
                
                x1 = np.array(image_batch).reshape(-1, N_INPUT)
                y1 = np.array(label_batch).reshape(-1, CLASSES)
                
                loss, opt, acc = sess.run([cost, optimizer, accuracy], feed_dict={x: x1, y: y1, keep_prob: DROPOUT})
                cost_periter.append(loss)
                acc_periter.append(acc)
                tmp_acc.append(acc)
                tmp_cost.append(loss)
            
            avg_acc = sum(tmp_acc)/len(tmp_acc)
            avg_cost = sum(tmp_cost)/len(tmp_cost) 
            toc = time.clock()    
            if iter % DISPLAY_STEP == 0:
                print(iter, " accuracy: %.5f" % avg_acc, "cost: %.5f" % avg_cost, "time: %.1f s" % (toc-tic))
                    
            
            

        # Testing Part
        for num in range(test_len):
            x2 = np.array(testimages[k][num]).reshape(-1, N_INPUT)
            y2 = np.array(testlabels[k][num]).reshape(-1, CLASSES)

            acc_test, prediction, labl = sess.run([accuracy, pred, lab], feed_dict = {x: x2, y: y2, keep_prob: 1.})
            # print(k, num, "test accuracy: %.5f" % acc_test, "prediction: %.5f" % prediction, "label: %.5f" % labl)
            test_acc.append(acc_test)
            test_prediction.append(prediction[0])
            test_label.append(labl[0])
        
        # sess.close()
        saver.save(sess, chkptfile)

    outlist = zip(cost_periter, acc_periter)
    filename = "train-%d.csv" % k
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(outlist)

    outtestlist = zip(test_acc, test_prediction, test_label)
    filename = "test-%d.csv" % k
    with open(filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(outtestlist)
    end = time.clock()
    runtime = end - start
    runtimes.append(runtime)
    print("time: %.2f s" % runtime)

with open("runtimes_normal.csv") as f:
    writer = csv.writer(f)
    writer.writerows(runtimes)
