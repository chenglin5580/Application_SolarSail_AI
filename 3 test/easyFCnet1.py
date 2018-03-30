import tensorflow as tf
import pandas as pd
import numpy as np

###定义变量
#训练集
TRAIN_SIZE=0.8
VALIDATION_SIZE=0.1
BATCH_SIZE=50
TRAIN_LIMIT=600000#训练循环次数


'''
数据读取阶段
'''
###读取数据
data = pd.read_csv("result0.csv")
#把灰度数据存储到images中
locations = data.iloc[:, 0:3].values #从第1列开始取数据，忽略第0列（label）
LOCS_SIZE=locations.shape[0]
# print("{0} {1}".format(imgSize,imgSize))
#把真实数字存到labels中
accelerations =data.iloc[:, 3].values
accelerations = np.multiply(accelerations, 100000)

#把数据划分为训练集、验证集和测试集
TRAIN_SIZE=int(0.8*LOCS_SIZE)
VALIDATION_SIZE=int(0.01*LOCS_SIZE)
train_images = locations[:TRAIN_SIZE]
train_labels = accelerations[:TRAIN_SIZE]
validation_images = locations[:(TRAIN_SIZE+VALIDATION_SIZE)]
validation_labels = accelerations[:(TRAIN_SIZE + VALIDATION_SIZE)]
test_images = locations[(TRAIN_SIZE+VALIDATION_SIZE):]
test_labels = accelerations[(TRAIN_SIZE + VALIDATION_SIZE):]


'''
神经网络的创建
全连接网络
'''
###进入TensorFlow部分
x = tf.placeholder("float", [None, 3])
y_true = tf.placeholder("float", [None, 1])

with tf.variable_scope('network'):
    h_fc1 = tf.layers.dense(x, units=200, activation=tf.nn.relu)
    h_fc2 = tf.layers.dense(h_fc1, units=200, activation=tf.nn.relu)
    y = tf.layers.dense(h_fc2, units=1, activation=tf.nn.relu)
tf.summary.histogram('network', y)


loss = tf.reduce_mean(tf.square(y_true - y))  #误差
tf.summary.scalar('loss', loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

accuracy = tf.reduce_sum(tf.square(y_true - y))

###初始化TensorFlow会话
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)




"""
开始训练
"""
###训练函数
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# sess.run(train_step, feed_dict={x: train_images, y_true: train_labels,keep_prob: 1.0})
# print(sess.run(accuracy, feed_dict={x: validation_images, y_true: validation_labels,keep_prob: 1.0}))
###训练数据流水线
#这个函数能够保证训练集不够时，重复运用
def next_batch(batch_size):
    # 要进行训练的图像
    global train_images
    #要进行训练的结果
    global train_labels
    #当前训练的样本数
    global index_in_epoch
    #epochs计数
    global epochs_completed

    #从0开始，每调用一次函数就增加50个样本
    start = index_in_epoch
    index_in_epoch += batch_size

    # 当所有的训练集都被使用时, 再重新随机排序
    if index_in_epoch > num_examples:
        #每全部训练完一次，epoch+1
        epochs_completed += 1
        #迭代完之后，打乱数据
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        #重新开始下一轮epochs
        start = 0
        index_in_epoch = batch_size
        #断言：如果batch_size超出了数据集的范围，则返回异常
        assert batch_size <= num_examples
    end = index_in_epoch
    #每50个取一次，同时取图像和结果
    return train_images[start:end], train_labels[start:end]


for i in range(TRAIN_LIMIT):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)#每50个数据为一个batch
    batch_ys = batch_ys[:, np.newaxis]
    if (i+1) % 100 == 0:
        loss11 = sess.run(loss, feed_dict={x: batch_xs, y_true: batch_ys})
        print(loss11)
        result_merge = sess.run(merged, feed_dict={x: batch_xs, y_true: batch_ys})
        writer.add_summary(result_merge, i)
        # train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys})
        # validation_accuracy = accuracy.eval(feed_dict={x: validation_images[0:BATCH_SIZE],y_true: validation_labels[0:BATCH_SIZE]})
        # validation_accuracy_total = accuracy.eval(feed_dict={x: validation_images, y_true: validation_labels,keep_prob: 1.0})
        # print('迭代次数:%d 训练集精确度:%.2f  总验证精确度:%.6f ' % (i+1, train_accuracy, validation_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})
    # train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys})

    # print(y.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0}))


'''
测试集
'''
# print('test')
# #按照test_images的结构来创建一个全为0的矩阵
# predicted_lables = np.zeros((test_images.shape[0],3))
# for i in range(0,test_images.shape[0]//BATCH_SIZE):
#     #和前面训练集一样，每50个样本进行一次操作
#     predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE,:] = y.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE],keep_prob: 1.0})
#
# # save results
# np.savetxt('test_result.csv',
#            np.c_[range(1,len(test_images)+1),predicted_lables],
#            delimiter=',',
#            header = 'ImageId,Label',
#            comments = '',
#            fmt='%d')
#
# sess.close()