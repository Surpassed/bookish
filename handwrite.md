#Bookish
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#加载数据集
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
import pylab


#创建图
tf.reset_default_graph()
#占位符（输入层）
x = tf.placeholder(tf.float32,[None,784])#数据集维度
y = tf.placeholder(tf.float32,[None,10])#数字0到9，10个类别
#定义学习参数
w = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))
#定义输出节点（softmax）
pred = tf.nn.softmax(tf.matmul(x,w)+b)
#反向传播结构
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))#误差值
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#迭代数
training_epochs = 30
batch_size = 100#每批数据量
display_step = 1#训练一次打印一次
#启动会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#初始化变量
    #开始训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        #循环所有数据集
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch([batch_size])
            c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
            avg_cost += c / total_batch
        #显示训练信息
        if (epoch+1) % display_step == 0:
            print("epoch:",'%04d'% (epoch+1),"cost=","{:9f}".format(avg_cost))
    #测试
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))#实际值与预测值比较
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy:",accuracy.eval({x:mnist.test,y:mnist.test.lables}))#打印准确率
    output = tf.argmax(pred,1)
    batch_xs,batch_ys = mnist.train.next_batch(2)
    outputval,predv = sess.run([output,pred],feed_dict={x:batch_xs})
    print(outputval,predv,batch_ys)

    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow()
    pylab.show()

