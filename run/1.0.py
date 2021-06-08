from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
import nibabel as nib
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


path = '/MICCAI_BraTS2020_TrainingData/'
path3 = '/MICCAI_BraTS2020_ValidationData/'
path1 = '/MICCAI_BraTS2019_TrainingData/'
path2 = '/MICCAI_BraTS2018_TrainingData/'

my_dir = sorted(os.listdir(path))
a = []
c = []
gt = []
for p in tqdm(my_dir):
    data_list = sorted(os.listdir(path + p))
    img_itk = sitk.ReadImage(path + p + '/' + data_list[0])
    flair = sitk.GetArrayFromImage(img_itk)
    img_itk = sitk.ReadImage(path + p + '/' + data_list[1])
    seg = sitk.GetArrayFromImage(img_itk)
    img_itk = sitk.ReadImage(path + p + '/' + data_list[4])
    t2 = sitk.GetArrayFromImage(img_itk)
    a.append(flair)
    c.append(t2)
    gt.append(seg)

flair = np.asarray(a, dtype=np.float32)
del a
pre_flair = flair[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_flair = flair[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del flair
for i in range(pre_flair.shape[0]):
    pre_flair[i, :, :, :] = pre_flair[i, :, :, :] / np.max(pre_flair[i, :, :, :])
pre_flair = pre_flair[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_flair.shape[0]):
    post_flair[i, :, :, :] = post_flair[i, :, :, :] / np.max(post_flair[i, :, :, :])
post_flair = post_flair[:, :, :, :].reshape([-1, 176, 144, 1])

t2 = np.asarray(c, dtype=np.float32)
del c
pre_t2 = t2[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_t2 = t2[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del t2
for i in range(pre_t2.shape[0]):
    pre_t2[i, :, :, :] = pre_t2[i, :, :, :] / np.max(pre_t2[i, :, :, :])
pre_t2 = pre_t2[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_t2.shape[0]):
    post_t2[i, :, :, :] = post_t2[i, :, :, :] / np.max(post_t2[i, :, :, :])
post_t2 = post_t2[:, :, :, :].reshape([-1, 176, 144, 1])

pre_x = np.concatenate((pre_flair, pre_t2), axis=3)
post_x = np.concatenate((post_flair, post_t2), axis=3)
y = np.asarray(gt, dtype=np.int8)
del gt
del pre_flair, pre_t2
del post_flair, post_t2
y_ini = y[:, 2:146, 32:208, 48:192].reshape([-1, 176, 144, 1])
y_pre = y[:, 1:145, 32:208, 48:192].reshape([-1, 176, 144, 1])
y_post = y[:, 3:147, 32:208, 48:192].reshape([-1, 176, 144, 1])
del y

my_dir1 = sorted(os.listdir(path1))
a = []
c = []
gt = []
for p in tqdm(my_dir1):
    data_list = sorted(os.listdir(path1 + p))
    img_itk = sitk.ReadImage(path1 + p + '/' + data_list[0])
    flair = sitk.GetArrayFromImage(img_itk)
    img_itk = sitk.ReadImage(path1 + p + '/' + data_list[1])
    seg = sitk.GetArrayFromImage(img_itk)
    img_itk = sitk.ReadImage(path1 + p + '/' + data_list[4])
    t2 = sitk.GetArrayFromImage(img_itk)
    a.append(flair)
    c.append(t2)
    gt.append(seg)

flair = np.asarray(a, dtype=np.float32)
del a
pre_flair = flair[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_flair = flair[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del flair
for i in range(pre_flair.shape[0]):
    pre_flair[i, :, :, :] = pre_flair[i, :, :, :] / np.max(pre_flair[i, :, :, :])
pre_flair = pre_flair[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_flair.shape[0]):
    post_flair[i, :, :, :] = post_flair[i, :, :, :] / np.max(post_flair[i, :, :, :])
post_flair = post_flair[:, :, :, :].reshape([-1, 176, 144, 1])

t2 = np.asarray(c, dtype=np.float32)
del c
t2_initial = t2[:, 2:146, 32:208, 48:192].reshape([-1, 144, 176, 144])
pre_t2 = t2[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_t2 = t2[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del t2

for i in range(pre_t2.shape[0]):
    pre_t2[i, :, :, :] = pre_t2[i, :, :, :] / np.max(pre_t2[i, :, :, :])
pre_t2 = pre_t2[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_t2.shape[0]):
    post_t2[i, :, :, :] = post_t2[i, :, :, :] / np.max(post_t2[i, :, :, :])
post_t2 = post_t2[:, :, :, :].reshape([-1, 176, 144, 1])

pre_x1 = np.concatenate((pre_flair, pre_t2), axis=3)
del pre_flair, pre_t2
post_x1 = np.concatenate((post_flair, post_t2), axis=3)
del post_flair, post_t2

pre_x = np.concatenate((pre_x, pre_x1), axis=0)
del pre_x1

post_x = np.concatenate((post_x, post_x1), axis=0)
del post_x1

y1 = np.asarray(gt, dtype=np.int8)
del gt

y_ini1 = y1[:, 2:146, 32:208, 48:192].reshape([-1, 176, 144, 1])
y_pre1 = y1[:, 1:145, 32:208, 48:192].reshape([-1, 176, 144, 1])
y_post1 = y1[:, 3:147, 32:208, 48:192].reshape([-1, 176, 144, 1])
del y1
y_ini = np.concatenate((y_ini, y_ini1), axis=0)
del y_ini1
y_pre = np.concatenate((y_pre, y_pre1), axis=0)
del y_pre1
y_post = np.concatenate((y_post, y_post1), axis=0)
del y_post1

my_dir2 = sorted(os.listdir(path2))
a = []
c = []
gt = []
for p in tqdm(my_dir2):
    data_list = sorted(os.listdir(path2 + p))
    img_itk = sitk.ReadImage(path2 + p + '/' + data_list[0])
    flair = sitk.GetArrayFromImage(img_itk)
    img_itk = sitk.ReadImage(path2 + p + '/' + data_list[1])
    seg = sitk.GetArrayFromImage(img_itk)
    img_itk = sitk.ReadImage(path2 + p + '/' + data_list[4])
    t2 = sitk.GetArrayFromImage(img_itk)
    a.append(flair)
    c.append(t2)
    gt.append(seg)

flair = np.asarray(a, dtype=np.float32)
del a
pre_flair = flair[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_flair = flair[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del flair
for i in range(pre_flair.shape[0]):
    pre_flair[i, :, :, :] = pre_flair[i, :, :, :] / np.max(pre_flair[i, :, :, :])
pre_flair = pre_flair[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_flair.shape[0]):
    post_flair[i, :, :, :] = post_flair[i, :, :, :] / np.max(post_flair[i, :, :, :])
post_flair = post_flair[:, :, :, :].reshape([-1, 176, 144, 1])

t2 = np.asarray(c, dtype=np.float32)
del c
pre_t2 = t2[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_t2 = t2[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del t2
for i in range(pre_t2.shape[0]):
    pre_t2[i, :, :, :] = pre_t2[i, :, :, :] / np.max(pre_t2[i, :, :, :])
pre_t2 = pre_t2[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_t2.shape[0]):
    post_t2[i, :, :, :] = post_t2[i, :, :, :] / np.max(post_t2[i, :, :, :])
post_t2 = post_t2[:, :, :, :].reshape([-1, 176, 144, 1])

pre_x2 = np.concatenate((pre_flair, pre_t2), axis=3)
del pre_flair, pre_t2
pre_x = np.concatenate((pre_x, pre_x2), axis=0)
del pre_x2
post_x2 = np.concatenate((post_flair, post_t2), axis=3)
del post_flair, post_t2
post_x = np.concatenate((post_x, post_x2), axis=0)
del post_x2
y2 = np.asarray(gt, dtype=np.int8)
del gt

y_ini2 = y2[:, 2:146, 32:208, 48:192].reshape([-1, 176, 144, 1])
y_pre2 = y2[:, 1:145, 32:208, 48:192].reshape([-1, 176, 144, 1])
y_post2 = y2[:, 3:147, 32:208, 48:192].reshape([-1, 176, 144, 1])

del y2
y_ini = np.concatenate((y_ini, y_ini2), axis=0)
del y_ini2
y_pre = np.concatenate((y_pre, y_pre2), axis=0)
del y_pre2
y_post = np.concatenate((y_post, y_post2), axis=0)
del y_post2

x = np.concatenate((pre_x, post_x), axis=3)
del pre_x, post_x
wt_ini = np.zeros(shape=y_ini.shape, dtype=np.int8)
wt_ini[np.where(y_ini > 0)] = 1
del y_ini
wt_pre = np.zeros(shape=y_pre.shape, dtype=np.int8)
wt_pre[np.where(y_pre > 0)] = 1
del y_pre
wt_post = np.zeros(shape=y_post.shape, dtype=np.int8)
wt_post[np.where(y_post > 0)] = 1
del y_post

y = np.concatenate((wt_ini, wt_pre, wt_post), axis=3)
print(y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15, random_state=0)
del x, y

X_pre = X_train[:, :, :, 0:2]
X_post = X_train[:, :, :, 2:4]
del X_train

X_test_pre = X_test[:, :, :, 0:2]
X_test_post = X_test[:, :, :, 2:4]
del X_test

Y_train = np.asarray(Y_train, dtype=np.float32)
Y_test = np.asarray(Y_test, dtype=np.float32)

Y_ini = Y_train[:, :, :, 0]
Y_ini = Y_ini.reshape([-1, 176, 144, 1])
Y_pre = Y_train[:, :, :, 1]
Y_pre = Y_pre.reshape([-1, 176, 144, 1])
Y_post = Y_train[:, :, :, 2]
Y_post = Y_post.reshape([-1, 176, 144, 1])
del Y_train

Y_ini_test = Y_test[:, :, :, 0]
Y_pre_test = Y_test[:, :, :, 1]
Y_post_test = Y_test[:, :, :, 2]
del Y_test

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():

    def se_block(input_tensor, c=16):
        num_channels = int(input_tensor.shape[-1])  # Tensorflow backend
        bottleneck = int(num_channels // c)

        se_branch = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
        se_branch = tf.keras.layers.Dense(bottleneck, use_bias=False, activation=tf.keras.activations.relu)(se_branch)
        se_branch = tf.keras.layers.Dropout(0.2)(se_branch)
        se_branch = tf.keras.layers.Dense(num_channels, use_bias=False, activation=tf.keras.activations.sigmoid)(se_branch)
        se_branch = tf.keras.layers.Dropout(0.2)(se_branch)

        out = tf.keras.layers.Multiply()([input_tensor, se_branch])
        return out

    def Global_Attention_Upsample(low_level_input, high_level_input, num_channels):

        low = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=(3, 3), padding='same')(low_level_input)
        high = tf.reduce_mean(high_level_input, axis=(-2, -3), keepdims=True)
        high = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=(1, 1), padding='same',activation='sigmoid')(high)
        mul = tf.keras.layers.Multiply()([low, high])
        up = tf.keras.layers.UpSampling2D(size=(2, 2))(high_level_input)
        up = tf.keras.layers.Conv2D(filters=num_channels, kernel_size=(1, 1), padding='same')(up)
        add = tf.keras.layers.Add()([up, mul])
        return add

    # 输入尺寸
    pre = tf.keras.Input(shape=(176, 144, 2), name="pre")
    post = tf.keras.Input(shape=(176, 144, 2), name="post")

    a2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(pre)
    a2 = tf.keras.layers.BatchNormalization()(a2)
    a2 = tf.keras.activations.relu(a2)
    a2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(a2)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(pre)
    a2 = tf.keras.layers.Add()([x, a2])

    s2 = se_block(a2)

    a3 = tf.keras.layers.BatchNormalization()(s2)
    a3 = tf.keras.activations.relu(a3)
    a3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2,2), padding='same')(a3)
    a3 = tf.keras.layers.BatchNormalization()(a3)
    a3 = tf.keras.activations.relu(a3)
    a3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(a3)
    s2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2,2), padding='same')(s2)
    a3 = tf.keras.layers.Add()([a3, s2])

    s3 = se_block(a3)

    a4 = tf.keras.layers.BatchNormalization()(s3)
    a4 = tf.keras.activations.relu(a4)
    a4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='same')(a4)
    a4 = tf.keras.layers.BatchNormalization()(a4)
    a4 = tf.keras.activations.relu(a4)
    a4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(a4)
    s3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same')(s3)
    a4 = tf.keras.layers.Add()([a4, s3])

    s4 = se_block(a4)

    a5 = tf.keras.layers.BatchNormalization()(s4)
    a5 = tf.keras.activations.relu(a5)
    a5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2,2), padding='same')(a5)
    a5 = tf.keras.layers.BatchNormalization()(a5)
    a5 = tf.keras.activations.relu(a5)
    a5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(a5)
    s4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same')(s4)
    a5 = tf.keras.layers.Add()([a5, s4])

    a200 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(a5)
    a200 = tf.keras.layers.UpSampling2D(size=(2, 2))(a200)
    con1 = tf.keras.layers.concatenate([a200, a4], axis=-1)
    a200 = tf.keras.layers.BatchNormalization()(con1)
    a200 = tf.keras.activations.relu(a200)
    a200 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(a200)
    a200 = tf.keras.layers.BatchNormalization()(a200)
    a200 = tf.keras.activations.relu(a200)
    a200 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(a200)
    con1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(con1)
    a200 = tf.keras.layers.Add()([a200, con1])

    a6 = Global_Attention_Upsample(a3, a200, 64)
    con1 = tf.keras.layers.concatenate([a6, a3], axis=-1)
    a6 = tf.keras.layers.BatchNormalization()(con1)
    a6 = tf.keras.activations.relu(a6)
    a6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(a6)
    a6 = tf.keras.layers.BatchNormalization()(a6)
    a6 = tf.keras.activations.relu(a6)
    a6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(a6)
    con1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(con1)
    a6 = tf.keras.layers.Add()([a6, con1])

    a7 = Global_Attention_Upsample(a2, a6, 32)
    con2 = tf.keras.layers.concatenate([a7, a2], axis=-1)
    a7 = tf.keras.layers.BatchNormalization()(con2)
    a7 = tf.keras.activations.relu(a7)
    a7 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(a7)
    a7 = tf.keras.layers.BatchNormalization()(a7)
    a7 = tf.keras.activations.relu(a7)
    a7 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(a7)
    con2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(con2)
    a7 = tf.keras.layers.Add()([a7, con2])

    end_pre = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                                     activation=tf.keras.activations.sigmoid, name="end_pre")(a7)

    # ####################################################################################################
    b2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(post)
    b2 = tf.keras.layers.BatchNormalization()(b2)
    b2 = tf.keras.activations.relu(b2)
    b2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(b2)
    b_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(post)
    b2 = tf.keras.layers.Add()([b2, b_2])

    bs2 = se_block(b2)

    b3 = tf.keras.layers.BatchNormalization()(bs2)
    b3 = tf.keras.activations.relu(b3)
    b3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2,2), padding='same')(b3)
    b3 = tf.keras.layers.BatchNormalization()(b3)
    b3 = tf.keras.activations.relu(b3)
    b3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(b3)
    b_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(2,2), padding='same')(bs2)
    b3 = tf.keras.layers.Add()([b3, b_3])

    bs3 = se_block(b3)

    b4 = tf.keras.layers.BatchNormalization()(bs3)
    b4 = tf.keras.activations.relu(b4)
    b4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='same')(b4)
    b4 = tf.keras.layers.BatchNormalization()(b4)
    b4 = tf.keras.activations.relu(b4)
    b4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(b4)
    b_4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2), padding='same')(bs3)
    b4 = tf.keras.layers.Add()([b4, b_4])

    bs4 = se_block(b4)

    b5 = tf.keras.layers.BatchNormalization()(bs4)
    b5 = tf.keras.activations.relu(b5)
    b5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2,2), padding='same')(b5)
    b5 = tf.keras.layers.BatchNormalization()(b5)
    b5 = tf.keras.activations.relu(b5)
    b5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')(b5)
    b_5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same')(bs4)
    b5 = tf.keras.layers.Add()([b5, b_5])

    b200 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(b5)
    b200 = tf.keras.layers.UpSampling2D(size=(2, 2))(b200)
    bcon1 = tf.keras.layers.concatenate([b200, b4], axis=-1)
    b200 = tf.keras.layers.BatchNormalization()(bcon1)
    b200 = tf.keras.activations.relu(b200)
    b200 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(b200)
    ba200 = tf.keras.layers.BatchNormalization()(b200)
    b200 = tf.keras.activations.relu(b200)
    b200 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(b200)
    bcon1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(bcon1)
    b200 = tf.keras.layers.Add()([b200, bcon1])

    b6 = Global_Attention_Upsample(b3, b200, 64)
    bcon1 = tf.keras.layers.concatenate([b6, b3], axis=-1)
    b6 = tf.keras.layers.BatchNormalization()(bcon1)
    b6 = tf.keras.activations.relu(b6)
    b6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(b6)
    b6 = tf.keras.layers.BatchNormalization()(b6)
    b6 = tf.keras.activations.relu(b6)
    b6 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(b6)
    bcon1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(bcon1)
    b6 = tf.keras.layers.Add()([b6, bcon1])

    b7 = Global_Attention_Upsample(b2, b6, 32)
    bcon2 = tf.keras.layers.concatenate([b7, b2], axis=-1)
    b7 = tf.keras.layers.BatchNormalization()(bcon2)
    b7 = tf.keras.activations.relu(b7)
    b7 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(b7)
    b7 = tf.keras.layers.BatchNormalization()(b7)
    b7 = tf.keras.activations.relu(b7)
    b7 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(b7)
    bcon2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(bcon2)
    b7 = tf.keras.layers.Add()([b7, bcon2])

    end_post = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                                      activation=tf.keras.activations.sigmoid, name="end_post")(b7)

    # ####################################################################################################

    c2add = tf.keras.layers.Add()([a7, b7])
    c2 = tf.keras.activations.relu(c2add)
    c2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(c2)
    c2 = tf.keras.activations.sigmoid(c2)
    c2 = tf.keras.layers.multiply([c2add, c2])

    c3add = tf.keras.layers.Add()([a6, b6])
    c3 = tf.keras.activations.relu(c3add)
    c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(c3)
    c3 = tf.keras.activations.sigmoid(c3)
    c3 = tf.keras.layers.multiply([c3add, c3])

    c4add = tf.keras.layers.Add()([a200, b200])
    c4 = tf.keras.activations.relu(c4add)
    c4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(c4)
    c4 = tf.keras.activations.sigmoid(c4)
    c4 = tf.keras.layers.multiply([c4add, c4])

    conc = tf.keras.layers.Add()([a5, b5])
    conc = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same')(conc)
    c5 = tf.keras.activations.relu(conc)
    c5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), padding='same')(c5)
    c5 = tf.keras.activations.sigmoid(c5)
    c5 = tf.keras.layers.multiply([conc, c5])

    upc_0 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(c5)
    upc_0 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(upc_0)
    concatc_1 = tf.keras.layers.Add()([upc_0, c4])
    concatc_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(concatc_1)

    c7 = tf.keras.layers.BatchNormalization()(concatc_1)
    c7 = tf.keras.activations.relu(c7)
    c7 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.activations.relu(c7)
    c7 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')(c7)
    c7 = tf.keras.layers.Add()([concatc_1, c7])

    upc_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(c7)
    upc_1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(upc_1)
    concatc_1 = tf.keras.layers.Add()([upc_1, c3])
    concatc_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(concatc_1)

    c8 = tf.keras.layers.BatchNormalization()(concatc_1)
    c8 = tf.keras.activations.relu(c8)
    c8 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.activations.relu(c8)
    c8 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(c8)
    c8 = tf.keras.layers.Add()([concatc_1, c8])

    upc_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(c8)
    upc_2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(upc_2)
    concatc_2 = tf.keras.layers.Add()([upc_2, c2])
    concatc_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(concatc_2)

    c9 = tf.keras.layers.BatchNormalization()(concatc_2)
    c9 = tf.keras.activations.relu(c9)
    c9 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.activations.relu(c9)
    c9 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(c9)
    c9 = tf.keras.layers.Add()([concatc_2, c9])

    end_ini = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same',
                                     activation=tf.keras.activations.sigmoid, name="end_ini")(c9)

    # dice系数
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
        return coef

    # 总损失函数，二分类交叉熵与dice_loss之和
    def loss(y_true, y_pred, cross_weights=0.4):
        def dice_coef_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

        return (tf.keras.losses.binary_crossentropy(y_true, y_pred) * cross_weights) + dice_coef_loss(y_true, y_pred)

    # 模型参数
    dsmodel = tf.keras.Model(inputs=[pre, post],
                             outputs=[end_pre, end_post, end_ini])
    dsmodel.summary()
    dsmodel.compile(loss=loss,
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=[dice_coef]
                    )


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


# 学习率衰减
def decay(epoch):
    if epoch < 10:
        return 1e-4
    elif epoch < 17:
        return 2e-5
    elif epoch < 24:
        return 1e-5
    elif epoch < 31:
        return 2e-6
    else:
        return 1e-6


# 在每个 epoch 结束时打印LR的回调（callbacks）。
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                          dsmodel.optimizer.lr.numpy()))


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True
                                       ),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

# 设置验证集比例为0.18
dsmodel.fit(x=[X_pre, X_post],
            y=[Y_pre, Y_post, Y_ini],
            batch_size=32,
            epochs=50,
            callbacks=callbacks,
            validation_split=0.18,
            shuffle=True
            )

dsmodel.save('./1.0.h5')
dsmodel.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
dsmodel.evaluate(x=[X_test_pre, X_test_post],
                 y=[Y_pre_test, Y_post_test, Y_ini_test])

my_dir = sorted(os.listdir(path))
gt = []
a = []
c = []
for p in tqdm(my_dir):
    data_list = sorted(os.listdir(path + p))
    img_itk = sitk.ReadImage(path + p + '/' + data_list[0])
    flair = sitk.GetArrayFromImage(img_itk)
    img_itk = sitk.ReadImage(path + p + '/' + data_list[3])
    t2 = sitk.GetArrayFromImage(img_itk)
    a.append(flair)
    c.append(t2)

flair = np.asarray(a, dtype=np.float32)
del a
pre_flair = flair[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_flair = flair[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del flair
for i in range(pre_flair.shape[0]):
    pre_flair[i, :, :, :] = pre_flair[i, :, :, :] / np.max(pre_flair[i, :, :, :])
pre_flair = pre_flair[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_flair.shape[0]):
    post_flair[i, :, :, :] = post_flair[i, :, :, :] / np.max(post_flair[i, :, :, :])
post_flair = post_flair[:, :, :, :].reshape([-1, 176, 144, 1])

t2 = np.asarray(c, dtype=np.float32)
del c
pre_t2 = t2[:, 1:145, 32:208, 48:192].reshape([-1, 144, 176, 144])
post_t2 = t2[:, 3:147, 32:208, 48:192].reshape([-1, 144, 176, 144])
del t2
for i in range(pre_t2.shape[0]):
    pre_t2[i, :, :, :] = pre_t2[i, :, :, :] / np.max(pre_t2[i, :, :, :])
pre_t2 = pre_t2[:, :, :, :].reshape([-1, 176, 144, 1])
for i in range(post_t2.shape[0]):
    post_t2[i, :, :, :] = post_t2[i, :, :, :] / np.max(post_t2[i, :, :, :])
post_t2 = post_t2[:, :, :, :].reshape([-1, 176, 144, 1])

pre_x = np.concatenate((pre_flair, pre_t2), axis=3)
post_x = np.concatenate((post_flair, post_t2), axis=3)

del pre_flair, pre_t2
del post_flair, post_t2


def write_hdf5(arr, fpath):
    with h5py.File(fpath, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


# 开始预测
pre_model = tf.keras.Model(
    inputs=dsmodel.inputs,
    outputs=dsmodel.get_layer('end_ini').output)
predictions = pre_model.predict(x=[pre_x, post_x],
                                batch_size=16
                                )
predictions = np.array(predictions)

print("predicted images size :")
print(predictions.shape)
print("保存预测的数据")
write_hdf5(predictions, "./predict.h5")
