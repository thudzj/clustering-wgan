from __future__ import print_function
from six.moves import xrange
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import mixture
from visualize import *
import svhn_data, cifar10_data
from dataset import make_dataset
import argparse
from metrics import cluster_acc, cluster_nmi

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak*x, x)

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def rescale(mat):
    return np.transpose(np.cast[np.float32]((-127.5 + mat)/127.5),(3,0,1,2))

parser = argparse.ArgumentParser('')
parser.add_argument('--data', type=str, default='mnist')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--logdir', type=str, default='')
parser.add_argument('--d', type=int, default=5)
parser.add_argument('--g', type=int, default=1)
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--wz', type=float, default=0.5)
parser.add_argument('--wx', type=float, default=0.1)
parser.add_argument('--sigma', type=float, default=0.1)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
batch_size = 64
z_dim = 128
device = '/gpu:0'
s = 32
Citers = args.d

image_dir = args.data
log_dir = './log_cwgan' + '/' + image_dir + '/' + args.logdir
ckpt_dir = './ckpt_cwgan' + '/' + image_dir + '/' + args.logdir
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if image_dir == 'mnist':
    channel = 1
    mnist = input_data.read_data_sets('./data/mnist', validation_size = 0)
    trainx = mnist.train._images
    trainy = mnist.train._labels.astype(np.int32)
    testx = mnist.test._images
    testy = mnist.test._labels.astype(np.int32)
    trainx = 2 * trainx - 1
    testx = 2 * testx - 1
    trainx = np.reshape(trainx, (-1, 28, 28, channel))
    testx = np.reshape(testx, (-1, 28, 28, channel))
    npad = ((0, 0), (2, 2), (2, 2), (0, 0))
    trainx = np.pad(trainx, pad_width=npad, mode='constant', constant_values=-1)
    testx = np.pad(testx, pad_width=npad, mode='constant', constant_values=-1)
elif image_dir == 'svhn':
    channel = 3
    trainx, trainy = svhn_data.load('./data/svhn','train')
    testx, testy = svhn_data.load('./data/svhn','test')
    trainx = rescale(trainx)
    testx = rescale(testx)
else:
    channel = 3
    trainx, trainy = cifar10_data.load("./data/cifar10", subset='train')
    testx, testy = cifar10_data.load("./data/cifar10", subset='test')
    trainx = np.transpose(trainx, [0, 2, 3, 1])
    testx = np.transpose(testx, [0, 2, 3, 1])

print(trainx.shape)
print(np.max(trainx), np.min(trainx))

def generator_conv(z, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()
        train = ly.fully_connected(z, 4 * 4 * 512, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm)
        train = tf.reshape(train, (-1, 4, 4, 512))
        train = ly.conv2d_transpose(train, 256, 5, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_params={'is_training':True})
        train = ly.conv2d_transpose(train, 128, 5, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_params={'is_training':True})
        train = ly.conv2d_transpose(train, channel, 5, stride=2,
                                    activation_fn=tf.nn.tanh, padding='SAME', normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_params={'is_training':True})
    return train

def critic_conv(x, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.conv2d(x, num_outputs=128, kernel_size=5,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=256, kernel_size=5,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=512, kernel_size=5,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        logit = ly.fully_connected(tf.reshape(img, [batch_size, -1]), 1, activation_fn=None)
    return logit

def critic_mlp(z, reuse=False):
    with tf.variable_scope('critic_mlp') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.fully_connected(z, 200, activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 200,activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 200,activation_fn=tf.nn.relu)
        logit = ly.fully_connected(img, 1, activation_fn=None)
    return logit

def encoder_z(x, reuse=False, flag = False):
    with tf.variable_scope('encoder_z') as scope:
        if reuse:
            scope.reuse_variables()
        if flag:
            img = ly.conv2d(tf.nn.dropout(x, 0.5), num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        else:
            img = ly.conv2d(x, num_outputs=64, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=128, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=256, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        img = ly.conv2d(img, num_outputs=512, kernel_size=3, stride=2, activation_fn=tf.nn.relu)
        logit = ly.fully_connected(tf.reshape(img, [batch_size, -1]), z_dim, activation_fn=None)
    return logit

def build_graph():
    generator = generator_conv
    critic = critic_conv
    critic_z = critic_mlp
    encoder = encoder_z

    unlabeled_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, 32, 32, channel))
    z = tf.placeholder(dtype=tf.float32, shape=(batch_size, z_dim))

    unlabeled_z = encoder(unlabeled_data)
    fake_unlabeled_data = generator(unlabeled_z)

    fake_x = generator(z, reuse=True)
    fake_z = encoder(fake_x, reuse=True)

    true_logit_1 = critic(unlabeled_data)
    true_logit_2 = critic_z(z)
    fake_logit_1 = critic(fake_x, reuse=True)
    fake_logit_2 = critic_z(unlabeled_z, reuse=True)

    c_loss_x = tf.reduce_mean(fake_logit_1 - true_logit_1)
    c_loss_z = tf.reduce_mean(fake_logit_2 - true_logit_2)

    recon_z = tf.losses.mean_squared_error(z, fake_z) #tf.reduce_sum(tf.abs(z - fake_z)) / batch_size
    recon_x = tf.losses.mean_squared_error(unlabeled_data, fake_unlabeled_data) #tf.reduce_sum(tf.abs(unlabeled_data - fake_unlabeled_data))
    e_loss = tf.reduce_mean(-fake_logit_2) + recon_z * args.wz + recon_x * args.wx
    g_loss = tf.reduce_mean(-fake_logit_1) + recon_z * args.wz + recon_x * args.wx

    e_loss_sum = tf.summary.scalar("e_loss", e_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_x_sum = tf.summary.scalar("c_loss_x", c_loss_x)
    c_loss_z_sum = tf.summary.scalar("c_loss_z", c_loss_z)

    img_sum = tf.summary.image("img", fake_x, max_outputs=10)

    theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_z')
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_mlp')
    theta_c_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    counter_c_z = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_z = ly.optimize_loss(loss=c_loss_z, learning_rate= None,
                    optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                    variables=theta_c_z, global_step=counter_c_z,
                    summaries = 'gradient_norm')

    counter_c_x = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_x = ly.optimize_loss(loss=c_loss_x, learning_rate= None,
                    optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                    variables=theta_c_x, global_step=counter_c_x,
                    summaries = 'gradient_norm')

    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=None,
                    optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                    variables=theta_g, global_step=counter_g,
                    summaries = 'gradient_norm')

    counter_e = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_e = ly.optimize_loss(loss=e_loss, learning_rate= None,
                    optimizer=tf.train.RMSPropOptimizer(learning_rate=5e-5),
                    variables=theta_e, global_step=counter_e,
                    summaries = 'gradient_norm')

    clipped_var_c_z = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in theta_c_z]
    with tf.control_dependencies([opt_c_z]):
        opt_c_z = tf.tuple(clipped_var_c_z)
        
    clipped_var_c_x = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in theta_c_x]
    with tf.control_dependencies([opt_c_x]):
        opt_c_x = tf.tuple(clipped_var_c_x)

    return unlabeled_data, z, opt_c_z, opt_c_x, opt_e, opt_g, fake_x, c_loss_z, c_loss_x, e_loss, g_loss, unlabeled_z


# In[9]:

def main():
    mus = np.concatenate([np.identity(10), np.zeros((10, z_dim-10))], 1)
    covs = np.stack([np.identity(z_dim) * args.sigma] * 10, 0)
    max_iter_step = 60000
    trainset = make_dataset(trainx, trainy)
    testset = make_dataset(testx, testy)
    with tf.device(device):
        unlabeled_data, z, opt_c_z, opt_c_x, opt_e, opt_g, fake_x, c_loss_z, c_loss_x, e_loss, g_loss, unlabeled_z = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    with tf.Session(config=config) as sess:
        def sample_z(gmm):
            if gmm is None:
                return np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
                ys = np.random.randint(10, size=batch_size)
                # np.random.multinomial(1, [1/10.]*10, size=batch_size)
                return np.concatenate([np.random.multivariate_normal(mus[y], covs[y], 1) for y in ys], 0)
            else:
                return gmm.sample(batch_size)[0]
        def sample_z_given_y(gmm, inx):
            if gmm is None:
                return np.random.normal(0, 1, [1, z_dim]).astype(np.float32)
                return np.random.multivariate_normal(mus[inx], covs[inx], 1)
            else:
                return np.random.multivariate_normal(gmm.means_[inx], gmm.covariances_[inx], 1)

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        gmm = None
        print("Training clustering wgan...")
        for i in range(max_iter_step):
            if i < 25 or i % 500 == 0:
                citers = 100
            else:
                citers = Citers

            for j in range(citers):
                if i % 100 == 99 and j == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, _, merged, loss_c_z_, loss_c_x_ = sess.run([opt_c_z, opt_c_x, merged_all, c_loss_z, c_loss_x], feed_dict={z: sample_z(gmm), unlabeled_data: trainset.next_batch(batch_size)[0]}, options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                else:
                    _, _ = sess.run([opt_c_z, opt_c_x], feed_dict={unlabeled_data: trainset.next_batch(batch_size)[0], z: sample_z(gmm)})

            if i % 100 == 99:
                _, _, merged, loss_e_, loss_g_ = sess.run([opt_g, opt_e, merged_all, e_loss, g_loss], feed_dict={z: sample_z(gmm), unlabeled_data: trainset.next_batch(batch_size)[0]}, options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'generator_and_encoder_metadata {}'.format(i), i)
            else:
                _, _ = sess.run([opt_g, opt_e], feed_dict={z: sample_z(gmm), unlabeled_data: trainset.next_batch(batch_size)[0]})

            if i % 100 == 99:
                print("Training ite %d, c_loss_z: %f, c_loss_x: %f, e_loss: %f, g_loss: %f" % (i, loss_c_z_, loss_c_x_, loss_e_, loss_g_))
                batch_y = []
                for j in range(10):
                    for k in range(10):
                        batch_y.append(j)
                batch_z = np.concatenate([sample_z_given_y(gmm, y) for y in batch_y[:64]], 0)
                bx = sess.run(fake_x, feed_dict={z: batch_z})
                batch_z = np.concatenate([sample_z_given_y(gmm, y) for y in batch_y[36:]], 0)
                bx1 = sess.run(fake_x, feed_dict={z: batch_z})
                bx = np.concatenate([bx, bx1[28:]], 0)

                fig = plt.figure(image_dir + '.clustering-wgan')
                grid_show(fig, (bx + 1) / 2, [32, 32, channel])
                if not os.path.exists('./logs/{}/{}'.format(image_dir, args.logdir)):
                    os.makedirs('./logs/{}/{}'.format(image_dir, args.logdir))
                fig.savefig('./logs/{}/{}/{}.png'.format(image_dir, args.logdir, (i-99)/100))

            if i % 500 == 0:
                trainset._index_in_epoch = 0
                true_zs = np.zeros((trainset.num_examples / batch_size * batch_size, z_dim))
                gts = np.zeros((trainset.num_examples / batch_size * batch_size))
                for j in range(trainset.num_examples / batch_size):
                    train_img, train_label = trainset.next_batch(batch_size)
                    bz = sess.run(unlabeled_z, feed_dict={unlabeled_data: train_img})
                    true_zs[j*batch_size:(j+1)*batch_size] = bz
                    gts[j*batch_size:(j+1)*batch_size] = train_label
                gmm = mixture.BayesianGaussianMixture(n_components=10, covariance_type='full').fit(true_zs)
                preds = gmm.predict(true_zs)
                print("Training ite %d, acc: %f, nmi: %f" % (i, cluster_acc(preds, gts), cluster_nmi(preds, gts)))
                #preds1 = np.argmax(np.sum(np.square(true_zs[:, np.newaxis, :] - mus[np.newaxis, :, :]), 2), 1)
                #print("Training ite %d, new gmm, acc: %f, nmi: %f; prior gmm, acc: %f, nmi: %f" % (i, cluster_acc(preds, gts), cluster_nmi(preds, gts), cluster_acc(preds1, gts), cluster_nmi(preds1, gts)))

            if i % 1000 == 999:
                saver.save(sess, os.path.join(
                    ckpt_dir, "model.ckpt"), global_step=i)

main()
