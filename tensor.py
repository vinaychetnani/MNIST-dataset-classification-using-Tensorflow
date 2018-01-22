from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

class load_idx:
    def __init__(self, file_name=None, fstream=None, file_handler=open):
        self.file_name = file_name
        self.fstream = fstream
        self.file_handler = file_handler
        self.magic_number = 0
        self.header_dtype = np.dtype(np.uint32).newbyteorder('>')
        if not (self.file_name is not None) ^ (self.fstream is not None):
            raise ValueError('Define either File Name or File Stream')
        elif self.file_name is not None:
            self.fstream = self.file_handler(self.file_name, 'rb')
    def get_magic_number(self):
        self.magic_number = np.frombuffer(self.fstream.read(4), dtype=self.header_dtype)[0]
        return self.magic_number
    def _extract_header(self):
        mask_dim = int('0x000000ff',16)
        mask_datatype = int('0x0000ff00',16)
        no_of_dimensions = np.bitwise_and(self.magic_number, np.array(mask_dim, dtype=np.uint32))
        datatype_index = np.right_shift(np.bitwise_and(self.magic_number, np.array(mask_datatype, dtype=np.uint32)),8)
        if datatype_index == int('0x08',16):
            dt = np.dtype(np.uint8)
        elif datatype_index == int('0x09',16):
            dt = np.dtype(np.int8)
        elif datatype_index == int('0x0B',16):
            dt = np.dtype(np.int16)
        elif datatype_index == int('0x0C',16):
            dt = np.dtype(np.int32)
        elif datatype_index == int('0x0D',16):
            dt = np.dtype(np.float32)
        elif datatype_index == int('0x0E',16):
            dt = np.dtype(np.float64)
        dimensions = np.empty(no_of_dimensions, dtype=np.uint32)
        for i in range(no_of_dimensions):
            read_val = np.frombuffer(self.fstream.read(4),dtype=self.header_dtype)
            dimensions[i] = read_val
        return dimensions, dt
    def load_file(self):
        if self.magic_number == 0:
            self.get_magic_number()
        [dimensions, dt] = self._extract_header()
        total_bytes_to_be_read = np.prod(dimensions, dtype=np.int32)*dt.itemsize
        data = np.frombuffer(self.fstream.read(total_bytes_to_be_read),dtype=dt)
        data = np.reshape(data,dimensions)
        if self.file_name is not None:
            self.fstream.close()
        return data

class load_mnist(load_idx):
    def __init__(self, file_name, file_type, file_handler=open, convert_to_float = False, display_sample = 0):
        load_idx.__init__(self, file_name = file_name, file_handler=file_handler)
        self.file_type = file_type
        self.convert_to_float = convert_to_float
        self.display_sample = display_sample
        self.mnist_magic_number={'data':2051, 'label':2049}
        if self.file_type == 'label':
            self.display_sample = 0
    def load(self):
        self.get_magic_number()
        if self.mnist_magic_number[self.file_type] == self.magic_number:
            self.data = self.load_file()
            if self.convert_to_float:
                self.data = self.data.astype(np.float32)
                self.data = np.multiply(self.data, 1.0/255.0)
            if self.display_sample != 0:
                self.display_samples(self.display_sample)
            return self.data
        else:
            print('Given file is not mnist : (%s,%s)'%(self.file_name, self.file_type))
    def display_samples(self, how_many=5):
        size = self.data.shape[0]
        perm = np.random.permutation(size)
        perm = perm[:how_many]
        images = self.data[perm,:,:]
        for i in range(how_many):
            fig = plt.figure()
            plt.imshow(images[i], cmap='Greys_r')
    def display_images(self, number):
        if number.shape.__len__() > 1:
            print('Number should be 1D array')
        else:
            for i in number:
                fig = plt.figure()
                plt.imshow(self.data[i], cmap='Greys_r')

training_set_file_name ='train_images_idx3_ubyte_customize.gz'
training_labels_file_name ='train_labels_idx1_ubyte_customize.gz'
testing_set_file_name ='test_images_idx3_ubyte_customize.gz'

train_images_obj=load_mnist(training_set_file_name, 'data', file_handler=gzip.GzipFile, display_sample=0)
train_labels_obj=load_mnist(training_labels_file_name, 'label', file_handler=gzip.GzipFile)
test_images_obj=load_mnist(testing_set_file_name, 'data', file_handler=gzip.GzipFile, display_sample=0)

train_images = train_images_obj.load()
train_labels = train_labels_obj.load()
test_images = test_images_obj.load()
train_images = train_images.reshape(train_images.shape[0],np.prod(train_images.shape[1:]))
test_images = test_images.reshape(test_images.shape[0], np.prod(test_images.shape[1:]))

print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
train_images2 = train_images[50000:60000, :]
train_images1 = train_images[0:50000, :]
t_label = train_labels.astype(float)
t_label1 = np.zeros((1,60000))
t_label1[0] = t_label
train_data1 = np.concatenate((t_label1.T, train_images), axis=1)


import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])




def run_training_op_on_data(op, data, kp, sess):
  return sess.run(op, feed_dict={
    x: data[:, 1:] / 255,
    y_: np.eye(10)[data[:, 0].astype(int)],
    keep_prob: kp
  })

W_conv1 = tf.Variable(tf.truncated_normal(([5, 5, 1, 32]), stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal(([5, 5, 32, 64]), stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = tf.Variable(tf.truncated_normal(([7 * 7 * 64, 1024]), stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal(([1024, 10]), stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

cur_index = 60000 + 1
for i in range(20000):
  if cur_index > 60000:
    permutation = np.arange(60000)
    np.random.shuffle(permutation)
    train_data1 = train_data1[permutation]
    cur_index = 0
  batch = train_data1[cur_index:cur_index+100]
  if i % 100 == 0:
    print("Step: {}, Accuracy: {}".format(i, run_training_op_on_data(accuracy, batch, 1.0, sess)))
  run_training_op_on_data(train_step, batch, 0.5, sess)
  cur_index += 100


test_results= np.empty([10000, 1])


for i in range(0,100):
 test_results[[z for z in xrange(i*100,(i+1)*100)],:] = sess.run(tf.argmax(y_conv, 1), feed_dict={x: test_images[[k for k in xrange(i*100,(i+1)*100)],:] / 255,keep_prob: 1.0}).reshape(100,1)
 print i

test_results=test_results.astype(int)
test_res = test_results.T[0]

np.savetxt("predicted_label_2014EE30542.csv", test_res, delimiter = ",")
