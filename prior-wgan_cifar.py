from __future__ import print_function
from six.moves import xrange
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import cluster
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
parser.add_argument('--data', type=str, default='cifar')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--logdir', type=str, default='')
parser.add_argument('--d', type=int, default=5)
parser.add_argument('--g', type=int, default=1)
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--wz', type=float, default=1)
parser.add_argument('--wx', type=float, default=1)
parser.add_argument('--wy', type=float, default=1)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--components', type=int, default=10)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--pretrain', type=int, default=60000)
parser.add_argument('--rprior', type=bool, default=True)
parser.add_argument('--renc', type=bool, default=False)
parser.add_argument('--rgen', type=bool, default=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
batch_size = args.batch_size
z_dim = args.z_dim
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
    trainx = np.concatenate([trainx, testx], 0)
    trainy = np.concatenate([trainy, testy], 0)
    trainx = (trainx - np.min(trainx)) / (np.max(trainx) - np.min(trainx))

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
                                    activation_fn=None, padding='SAME')#, normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_params={'is_training':True})
    return train

def critic_conv(x, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.conv2d(x, num_outputs=64, kernel_size=5,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=128, kernel_size=5,
                        stride=2, activation_fn=lrelu)#, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=256, kernel_size=5,
                        stride=2, activation_fn=lrelu)#, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=512, kernel_size=4,
                        stride=2, activation_fn=lrelu)#, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        logit = ly.fully_connected(tf.reshape(img, [batch_size, -1]), 1, activation_fn=None)
    return logit

def critic_mlp(z, reuse=False):
    with tf.variable_scope('critic_mlp') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.fully_connected(z, 256, activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 256,activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 256,activation_fn=tf.nn.relu)
        logit = ly.fully_connected(img, 1, activation_fn=None)
    return logit

def critic_mlp1(z, reuse=False):
    with tf.variable_scope('critic_mlp1') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.fully_connected(z, 256, activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 256,activation_fn=tf.nn.relu)
        img = ly.fully_connected(img, 256,activation_fn=tf.nn.relu)
        logit = ly.fully_connected(img, 1, activation_fn=None)
    return logit

def encoder_z(x, reuse=False):
    with tf.variable_scope('encoder_z') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.conv2d(x, num_outputs=64, kernel_size=3, stride=2, activation_fn=lrelu)#, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=128, kernel_size=3, stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=256, kernel_size=3, stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=512, kernel_size=3, stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.fully_connected(tf.reshape(img, [batch_size, -1]), 1024, activation_fn=lrelu, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = tf.nn.dropout(img, 0.7)
        logit = ly.fully_connected(img, z_dim, activation_fn=None)
    return logit

def prior(y, reuse=False):
    with tf.variable_scope('prior') as scope:
        if reuse:
            scope.reuse_variables()
        noise = tf.random_normal([batch_size, args.z_dim])
        output = ly.fully_connected(tf.concat([noise, y], 1), 256, activation_fn=tf.nn.relu)
        output = ly.fully_connected(output, 256, activation_fn=tf.nn.relu)
        output = ly.fully_connected(output, 256, activation_fn=tf.nn.relu)
        output = tf.nn.dropout(output, 0.7)
        output = ly.fully_connected(output, args.z_dim, activation_fn=None)
    return output

def encoder_y(z, reuse=False):
    with tf.variable_scope('encoder_y') as scope:
        if reuse:
            scope.reuse_variables()
        output = ly.fully_connected(z, 256, activation_fn=tf.nn.relu)
        output = ly.fully_connected(output, 256, activation_fn=tf.nn.relu)
        output = ly.fully_connected(output, 256, activation_fn=tf.nn.relu)
        output = tf.nn.dropout(output, 0.7)
        output = ly.fully_connected(output, 10, activation_fn=None)
    return output

def build_graph():
    generator = generator_conv
    critic = critic_conv
    critic_z_1 = critic_mlp
    critic_z_2 = critic_mlp1
    encoder = encoder_z

    unlabeled_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, 32, 32, channel))
    y = tf.placeholder(dtype=tf.float32, shape=(batch_size, 10))

    z1 = prior(y)
    fake_y = encoder_y(z1)

    z2 = encoder(unlabeled_data)
    fake_unlabeled_data = generator(z2)
    fake_x = generator(z1, reuse=True)

    logit_z1 = critic_z_1(z1)
    logit_z2 = critic_z_1(z2, reuse=True)

    c_loss_1 = tf.reduce_mean(logit_z2 - logit_z1)

    alpha = tf.random_uniform(
        shape=[batch_size, 1],
        minval=0.,
        maxval=1.
    )
    interpolates = z1 + (alpha * (z2 - z1))
    gradients = tf.gradients(critic_z_1(interpolates, reuse=True), [interpolates])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    c_loss_1 += (args.beta)*gradient_penalty

    logit_z1_2 = critic_z_2(z1)
    logit_z2_2 = critic_z_2(z2, reuse=True)
    c_loss_2 = tf.reduce_mean(logit_z1_2 - logit_z2_2)
    alpha1 = tf.random_uniform(
        shape=[batch_size, 1],
        minval=0.,
        maxval=1.
    )
    interpolates1 = z2 + (alpha1 * (z1 - z2))
    gradients1 = tf.gradients(critic_z_2(interpolates1, reuse=True), [interpolates1])
    slopes1 = tf.sqrt(tf.reduce_sum(tf.square(gradients1[0]), reduction_indices=[1]))
    gradient_penalty1 = tf.reduce_mean((slopes1-1.)**2)
    c_loss_2 += (args.beta)*gradient_penalty1

    recon_y = tf.losses.softmax_cross_entropy(logits=fake_y, onehot_labels=y)
    #tf.reduce_sum(tf.abs(unlabeled_data - fake_unlabeled_data))
    recon_x = tf.losses.sigmoid_cross_entropy(multi_class_labels=unlabeled_data, logits=fake_unlabeled_data)

    e_loss = recon_x * args.wx + tf.reduce_mean(-logit_z2)
    g_loss = recon_x

    prior_loss = recon_y * args.wy + tf.reduce_mean(-logit_z1_2)
    e_loss_y = recon_y

    e_loss_sum = tf.summary.scalar("e_loss", e_loss)
    e_loss_y_sum = tf.summary.scalar("e_loss_y", e_loss_y)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    prior_loss_sum = tf.summary.scalar("prior_loss", prior_loss)
    c_loss_1_sum = tf.summary.scalar("c_loss_1", c_loss_1)
    c_loss_2_sum = tf.summary.scalar("c_loss_2", c_loss_2)
    img_sum = tf.summary.image("img", fake_x, max_outputs=10)

    theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_z')
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    theta_e_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_y')
    theta_prior = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='prior')

    theta_c_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_mlp')
    theta_c_z_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_mlp1')

    pretrain_x = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5, beta2=0.9).minimize(recon_x, var_list=theta_e + theta_g)
    pretrain_y = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5, beta2=0.9).minimize(recon_y, var_list=theta_e_y + theta_prior)

    counter_c_1 = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_1 = ly.optimize_loss(loss=c_loss_1, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_c_z, global_step=counter_c_1,
                    summaries = 'gradient_norm')

    counter_c_2 = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_2 = ly.optimize_loss(loss=c_loss_2, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_c_z_2, global_step=counter_c_2,
                    summaries = 'gradient_norm')

    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_g, global_step=counter_g,
                    summaries = 'gradient_norm')

    counter_e = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_e = ly.optimize_loss(loss=e_loss, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_e, global_step=counter_e,
                    summaries = 'gradient_norm')

    counter_prior = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_prior = ly.optimize_loss(loss=prior_loss, learning_rate=None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_prior, global_step=counter_prior,
                    summaries = 'gradient_norm')

    counter_e_y = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_e_y = ly.optimize_loss(loss=e_loss_y, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_e_y, global_step=counter_e_y,
                    summaries = 'gradient_norm')

    return unlabeled_data, opt_c_1, opt_c_2, opt_e, opt_g, tf.nn.sigmoid(fake_x), c_loss_1, c_loss_2, e_loss, g_loss, opt_prior, prior_loss, opt_e_y, e_loss_y, y, pretrain_x, pretrain_y, tf.nn.sigmoid(fake_unlabeled_data), z2


# In[9]:

def main():

    max_iter_step = 60000
    al_iters = 2000
    trainset = make_dataset(trainx, trainy)
    testset = make_dataset(testx, testy)
    with tf.device(device):
        unlabeled_data, opt_c_1, opt_c_2, opt_e, opt_g, fake_x, c_loss_1, c_loss_2, e_loss, g_loss, opt_prior, prior_loss, opt_e_y, e_loss_y, y, pretrain_x, pretrain_y, fake_unlabeled_data, z2 = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        print("Pretraining unsupervised wgan...")
        try:
            saver.restore(sess, os.path.join(ckpt_dir, "pretrain.ckpt-59999"))
        except:
            for j in range(args.pretrain):
                _, _, loss_ae_y_, loss_ae_x_ = sess.run([pretrain_x, pretrain_y, e_loss_y, g_loss], feed_dict={y:np.random.multinomial(1, [1/10.]*10, size=batch_size), unlabeled_data: trainset.next_batch(batch_size)[0]})
                if j % 500 == 499:
                    print("Pretraining ite %d, ae_loss_y: %f, ae_loss_x: %f" % (j , loss_ae_y_, loss_ae_x_))
                    bx = sess.run(fake_unlabeled_data, feed_dict={unlabeled_data:trainset.next_batch(batch_size)[0]})
                    bx1 = sess.run(fake_unlabeled_data, feed_dict={unlabeled_data:trainset.next_batch(batch_size)[0]})
                    bx = np.concatenate([bx, bx1[2*batch_size-100:batch_size]], 0)
                    fig = plt.figure(image_dir + '.clustering-wgan')
                    grid_show(fig, bx, [32, 32, channel])
                    if not os.path.exists('./logs/{}/pretrain/{}'.format(image_dir, args.logdir)):
                        os.makedirs('./logs/{}/pretrain/{}'.format(image_dir, args.logdir))
                    fig.savefig('./logs/{}/pretrain/{}/{}.png'.format(image_dir, args.logdir, (j-499)/500))
                if j % 10000 == 9999:
                    saver.save(sess, os.path.join(ckpt_dir, "pretrain.ckpt"), global_step=j)

        print("Training unsupervised wgan...")
        
        for i in range(max_iter_step):

            for j in range(Citers):
                if i % 100 == 99 and j == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, _, merged, loss_c_1_, loss_c_2_ = sess.run([opt_c_1, opt_c_2, merged_all, c_loss_1, c_loss_2], feed_dict={unlabeled_data: trainset.next_batch(batch_size)[0], y:np.random.multinomial(1, [1/10.]*10, size=batch_size)}, options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                else:
                    _, _ = sess.run([opt_c_1, opt_c_2], feed_dict={y:np.random.multinomial(1, [1/10.]*10, size=batch_size),unlabeled_data: trainset.next_batch(batch_size)[0]})

            if i % 100 == 99:
                _, _, _, _, merged, loss_e_, loss_g_, loss_e_y_, loss_prior_ = sess.run([opt_g, opt_e, opt_prior, opt_e_y, merged_all, e_loss, g_loss, e_loss_y, prior_loss], feed_dict={y:np.random.multinomial(1, [1/10.]*10, size=batch_size), unlabeled_data: trainset.next_batch(batch_size)[0]}, options=run_options, run_metadata=run_metadata)
                print("Training ite %d, c_loss_1: %f, c_loss_2: %f, e_loss: %f, g_loss: %f, prior_loss: %f, e_loss_y: %f" % (i, loss_c_1_, loss_c_2_, loss_e_, loss_g_, loss_prior_, loss_e_y_))
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'ae_metadata {}'.format(i), i)
            else:
                _, _, _, _ = sess.run([opt_g, opt_e, opt_prior, opt_e_y], feed_dict={y:np.random.multinomial(1, [1/10.]*10, size=batch_size), unlabeled_data: trainset.next_batch(batch_size)[0]})

            if i % 100 == 99:

                batch_y = []
                for j in range(10):
                    for k in range(10):
                        batch_y.append(j)
                batch_y = dense_to_one_hot(np.asarray(batch_y))
                bx = sess.run(fake_x, feed_dict={y: batch_y[:batch_size]})
                bx1 = sess.run(fake_x, feed_dict={y: batch_y[100-batch_size:]})
                bx = np.concatenate([bx, bx1[2*batch_size-100:batch_size]], 0)
                fig = plt.figure(image_dir + '.clustering-wgan')
                grid_show(fig, bx, [32, 32, channel])
                if not os.path.exists('./logs/{}/{}'.format(image_dir, args.logdir)):
                    os.makedirs('./logs/{}/{}'.format(image_dir, args.logdir))
                fig.savefig('./logs/{}/{}/{}.png'.format(image_dir, args.logdir, (i-99)/100))

            if i % 1000 == 999:
                trainset.shuffle()
                true_zs = np.zeros((trainset._num_examples / batch_size * batch_size, z_dim))
                gts = np.zeros((trainset._num_examples / batch_size * batch_size))
                for j in range(trainset._num_examples / batch_size):
                    train_img, train_label = trainset.next_batch(batch_size)
                    bz = sess.run(z2, feed_dict={unlabeled_data: train_img})
                    true_zs[j*batch_size:(j+1)*batch_size] = bz
                    gts[j*batch_size:(j+1)*batch_size] = train_label
                preds = cluster.KMeans(n_clusters=10, n_jobs=-1).fit_predict(true_zs)
                print("Training ite %d, acc: %f, nmi: %f" % (i, cluster_acc(preds, gts), cluster_nmi(preds, gts)))

                saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=i)


main()
