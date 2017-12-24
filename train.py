# CUDA_VISIBLE_DEVICES='0' python train.py
import argparse
import numpy as np ; print 'numpy ' + np.__version__
import tensorflow as tf ; print 'tensorflow ' + tf.__version__
import cv2 ; print 'cv2 ' + cv2.__version__

# parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
args = parser.parse_args()
print args

x = tf.placeholder('float32', [1,256,256,1],name='x') ; print x
y = tf.placeholder('float32', [1,256,256,1],name='y') ; print y

n = tf.identity(x)
for i in range(20):
    n = tf.layers.conv2d(inputs=n,filters=1,kernel_size=3,padding='same',dilation_rate=1,activation=tf.nn.elu,reuse=tf.AUTO_REUSE,name='rule') ; print n
