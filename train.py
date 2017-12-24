# wget https://s3.amazonaws.com/nist-srd/SD4/NISTSpecialDatabase4GrayScaleImagesofFIGS.zip
# CUDA_VISIBLE_DEVICES='0' python train.py
import argparse
import numpy as np ; print 'numpy ' + np.__version__
import tensorflow as tf ; print 'tensorflow ' + tf.__version__
import cv2 ; print 'cv2 ' + cv2.__version__

# parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--png',help='png goal',default='./NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0/s0044_03.png')
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
args = parser.parse_args()
print args

x = tf.placeholder('float32', [1,256,256,1],name='x') ; print x
y = tf.placeholder('float32', [1,256,256,1],name='y') ; print y

with tf.variable_scope("rule"):
    n = tf.layers.conv2d(inputs=x,filters=1,kernel_size=3,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
    for i in range(40):
        n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=3,padding='same',dilation_rate=1,activation=tf.nn.elu,reuse=True) ; print n

loss = tf.reduce_mean(tf.squared_difference(n,y))
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
grads = opt.compute_gradients(loss)
train = opt.apply_gradients(grads)
norm = tf.global_norm([i[0] for i in grads])
init = tf.global_variables_initializer()

seed = np.zeros([1,256,256,1])
seed[0,128,128,0]=1

goal = cv2.imread(args.png)
goal = cv2.cvtColor(goal, cv2.COLOR_BGR2GRAY)
goal = cv2.resize(goal,(256,256))
goal = np.reshape(goal,[1,256,256,1])
goal = goal / 255.

print 'seed.shape',seed.shape,'goal.shape',goal.shape

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        # TRAIN
        larr=[] # losses
        narr=[] # gradients
        _,l_,n_ = sess.run([train,loss,norm],feed_dict={x:seed,y:goal})
        larr.append(l_)
        narr.append(n_)
        print 'epoch {:6d} loss {:12.8f} grad {:12.8f}'.format(i,np.mean(larr),np.mean(narr))

        # TEST
        if i%100==0:
            pred = sess.run(n, feed_dict={x:seed})
            cv2.imshow('img', cv2.resize(np.multiply(np.concatenate([seed[0],pred[0],goal[0]],axis=1),255.).astype(np.uint8),dsize=(0,0),fx=4,fy=4,interpolation=cv2.INTER_LANCZOS4))
            if cv2.waitKey(10) > 255:
                exit()
