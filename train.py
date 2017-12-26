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
parser.add_argument('--steps', help='steps to run automata', default=100, type=int)
parser.add_argument('--dilation', help='convolution dilation factor', default=1, type=int)
parser.add_argument('--kernel', help='convolution kernel size', default=3, type=int)
parser.add_argument('--rules', help='rules', default=3, type=int)
args = parser.parse_args()
print args

x = tf.placeholder('float32', [1,256,256,1],name='x') ; print x
y = tf.placeholder('float32', [1,256,256,1],name='y') ; print y

with tf.variable_scope("rule"):
    names=[]
    n = tf.layers.conv2d(inputs=x,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='0') ; print n
    for i in range(1,args.rules):    
        n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name=str(i)) ; print n
        for j in range(args.steps):
            n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name=str(i),reuse=True);print n


#    n = tf.layers.conv2d(inputs=x,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='1') ; print n
#    for i in range(args.steps):
#        n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='1',reuse=True) ; print n
#    n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='2') ; print n
#    for i in range(args.steps):
#        n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='2',reuse=True) ; print n
#    n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='3') ; print n
#    for i in range(args.steps):
#        n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='3',reuse=True) ; print n
#    n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='4') ; print n
#    for i in range(args.steps):
#        n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='4',reuse=True) ; print n
#    n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='5') ; print n
#    for i in range(args.steps):
#        n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=args.kernel,padding='same',dilation_rate=args.dilation,activation=tf.nn.elu,name='5',reuse=True) ; print n

loss = tf.reduce_mean(tf.squared_difference(n,y))
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
grads = opt.compute_gradients(loss)
train = opt.apply_gradients(grads)
norm = tf.global_norm([i[0] for i in grads])
init = tf.global_variables_initializer()

seed = np.zeros([1,256,256,1])
#seed[0,126:130,126:130,0]=[0.09729081,0.17524717,0.29401641,0.46918393],[0.87818577,0.03318266,0.05943777,0.3967884],[0.4674391,0.06020691,0.21784802,0.59732923],[0.47063308,0.89000737,0.64447307,0.83497568]
seed[0,126:130,126:130,0]=[0.34527012,0.5850928,0.21546599,0.74736086],[0.79759411,0.63966193,0.47159817,0.75842557],[0.65230226,0.2722718,0.25583924,0.58976751],[0.16484875,0.64111161,0.55920788,0.40978965]

goal = cv2.imread(args.png)
print type(goal)
goal = cv2.cvtColor(goal, cv2.COLOR_BGR2GRAY)
print type(goal)
goal = cv2.resize(goal,(256,256))
goal = np.reshape(goal,[1,256,256,1])
goal = goal / 255.
#print goal
#exit(0)
#goal = np.around(goal)
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
#            print pred[0]
            cv2.imshow('img', cv2.resize(np.multiply(np.concatenate([seed[0],pred[0],goal[0]],axis=1),255.).astype(np.uint8),dsize=(0,0),fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4))
#            cv2.imshow('img', cv2.resize(np.concatenate([seed[0],pred[0],goal[0]],axis=1).astype(np.uint8),dsize=(0,0),fx=2,fy=2,interpolation=cv2.INTER_LANCZOS4))
            cv2.waitKey(100)
