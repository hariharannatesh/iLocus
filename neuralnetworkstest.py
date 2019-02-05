import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors,svm

df=pd.read_csv('Databaseneural2.txt')
X=np.array(df.drop(['Cell1','Cell2','Cell3','Cell4','Cell5','Cell6','Cell7','Cell8','Cell9'],1))
#print(X)
Y=np.array(df[['Cell1','Cell2','Cell3','Cell4','Cell5','Cell6','Cell7','Cell8','Cell9']])
#print(Y)
learning_rate=0.001
epochs=30
batch_size=30
x=tf.placeholder(tf.float32,[None,4])
y=tf.placeholder(tf.float32,[None,9])
def neural_network_model(data):
    W1=tf.Variable(tf.random_normal([4,20],stddev=0.03),name='W1')
    b1=tf.Variable(tf.random_normal([20]),name='b1')
    W2=tf.Variable(tf.random_normal([20,30],stddev=0.03),name='W2')
    b2=tf.Variable(tf.random_normal([30]),name='b2')
    W3=tf.Variable(tf.random_normal([30,20],stddev=0.03),name='W3')
    b3=tf.Variable(tf.random_normal([20]),name='b3')
    W4=tf.Variable(tf.random_normal([20,9],stddev=0.03),name='W4')
    b4=tf.Variable(tf.random_normal([9]),name='b4')
    #W5=tf.Variable(tf.random_normal([10,9],stddev=0.03),name='W5')
    #b5=tf.Variable(tf.random_normal([9]),name='b5')
    hidden_out=tf.nn.relu(tf.add(tf.matmul(data,W1),b1))
    hidden_out=tf.nn.relu(tf.add(tf.matmul(hidden_out,W2),b2))
    hidden_out=tf.nn.relu(tf.add(tf.matmul(hidden_out,W3),b3))
    #hidden_out=tf.nn.relu(tf.add(tf.matmul(hidden_out,W4),b4))
    y_=tf.nn.softmax(tf.add(tf.matmul(hidden_out,W4),b4))
    return y_


def train_neural_network(x):
    prediction=neural_network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    X_train,X_test,y_train,y_test=model_selection.train_test_split(X,Y,test_size=0.3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_batch=int(len(X_train)/batch_size)
        print("Hi")
        for epoch in range(epochs):
            avg_cost=0
            for _ in range(100):

                for i in range(total_batch):
                    batch_x,batch_y= X_train[i*batch_size:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size]
                    _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                    avg_cost=(avg_cost+c)/total_batch
            print("Epoch:",(epoch+1),"cost=","{:.3f}".format(avg_cost))

        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))

        print(accuracy.eval({x:X_test,y:y_test}))
    
train_neural_network(x)
    

