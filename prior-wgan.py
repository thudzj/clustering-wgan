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
parser.add_argument('--data', type=str, default='mnist')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--logdir', type=str, default='')
parser.add_argument('--d', type=int, default=5)
parser.add_argument('--g', type=int, default=1)
parser.add_argument('--beta', type=float, default=10)
parser.add_argument('--wz', type=float, default=1)
parser.add_argument('--wx', type=float, default=0.1633)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--components', type=int, default=10)
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--tau0', type=float, default=1.0)
parser.add_argument('--hard', type=bool, default=False)
parser.add_argument('--trainpi', type=bool, default=True)
parser.add_argument('--rprior', type=bool, default=False)
parser.add_argument('--renc', type=bool, default=False)
parser.add_argument('--rgen', type=bool, default=False)

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
                                    activation_fn=tf.nn.tanh, padding='SAME')#, normalizer_fn=ly.batch_norm, weights_initializer=tf.random_normal_initializer(0, 0.02),normalizer_params={'is_training':True})
    return train

def critic_conv(x, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()
        img = ly.conv2d(x, num_outputs=128, kernel_size=5,
                        stride=2, activation_fn=lrelu)
        img = ly.conv2d(img, num_outputs=256, kernel_size=5,
                        stride=2, activation_fn=lrelu)#, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
        img = ly.conv2d(img, num_outputs=512, kernel_size=5,
                        stride=2, activation_fn=lrelu)#, normalizer_fn=ly.batch_norm, normalizer_params={'is_training':True})
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

def prior(noise, y, reuse=False):
    with tf.variable_scope('prior') as scope:
        if reuse:
            scope.reuse_variables()
        output = ly.fully_connected(tf.concat([noise, y], 1), args.z_dim, activation_fn=tf.nn.relu)
        output = ly.fully_connected(output, args.z_dim, activation_fn=tf.nn.relu)
        output = ly.fully_connected(output, args.z_dim, activation_fn=tf.nn.relu)
        output = ly.fully_connected(output, args.z_dim, activation_fn=None)
    return output

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

def build_graph():
    generator = generator_conv
    critic = critic_conv
    critic_z = critic_mlp
    encoder = encoder_z

    unlabeled_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, 32, 32, channel))

    noise = tf.random_normal([batch_size, args.z_dim])
    prior_z = []
    for i in range(10):
        y_ = np.zeros((batch_size, 10))
        y_[:][i] = 1
        y = tf.constant(y_.astype(np.float32))
        if i == 0:
            prior_z.append(prior(noise, y))
        else:
            prior_z.append(prior(noise, y, reuse=True))
    prior_z = tf.stack(prior_z, axis=1)
    # temperature
    tau = tf.Variable(5.0,name="temperature")
    with tf.variable_scope('prior') as scope:
        pi = tf.get_variable("pi", shape=[1, 10], initializer=tf.constant_initializer(value=1, dtype=tf.float32), trainable=args.trainpi)
    # set hard=True for ST Gumbel-Softmax
    sample_pi = gumbel_softmax(tf.tile(pi, [args.batch_size, 1]),tau,hard=args.hard)
    z = tf.reduce_sum(sample_pi[:, :, None] * prior_z, 1)

    unlabeled_z = encoder(unlabeled_data)
    fake_unlabeled_data = generator(unlabeled_z)

    fake_x = generator(z, reuse=True)
    fake_z = encoder(fake_x, reuse=True)

    true_logit_1 = critic(unlabeled_data)
    true_logit_2 = critic_z(unlabeled_z)
    fake_logit_1 = critic(fake_x, reuse=True)
    fake_logit_2 = critic_z(z, reuse=True)

    c_loss_x = tf.reduce_mean(fake_logit_1 - true_logit_1)
    c_loss_z = tf.reduce_mean(fake_logit_2 - true_logit_2)

    alpha = tf.random_uniform(
        shape=[batch_size, 1, 1, 1],
        minval=0.,
        maxval=1.
    )
    interpolates = unlabeled_data + (alpha * (fake_x - unlabeled_data))
    gradients = tf.gradients(critic(interpolates, reuse=True), [interpolates])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    c_loss_x += (args.beta)*gradient_penalty

    alpha1 = tf.random_uniform(
        shape=[batch_size, 1],
        minval=0.,
        maxval=1.
    )
    interpolates1 = unlabeled_z + (alpha1 * (z - unlabeled_z))
    gradients1 = tf.gradients(critic_z(interpolates1, reuse=True), [interpolates1])
    slopes1 = tf.sqrt(tf.reduce_sum(tf.square(gradients1[0]), reduction_indices=[1]))
    gradient_penalty1 = tf.reduce_mean((slopes1-1.)**2)
    c_loss_z += (args.beta)*gradient_penalty1

    recon_z = tf.losses.mean_squared_error(labels=z, predictions=fake_z) #tf.reduce_sum(tf.abs(z - fake_z)) / batch_size
    recon_x = tf.losses.mean_squared_error(labels=unlabeled_data, predictions=fake_unlabeled_data) #tf.reduce_sum(tf.abs(unlabeled_data - fake_unlabeled_data))
    if args.renc:
        e_loss = recon_x
    else:
        e_loss = recon_z * args.wz + recon_x * args.wx
    if args.rgen:
        g_loss = tf.reduce_mean(-fake_logit_1)
    else:
        g_loss = tf.reduce_mean(-fake_logit_1) + recon_z * args.wz + recon_x * args.wx
    if args.rprior:
        prior_loss = tf.reduce_mean(-fake_logit_2)
    else:
        prior_loss = tf.reduce_mean(-fake_logit_1-fake_logit_2)

    e_loss_sum = tf.summary.scalar("e_loss", e_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_x_sum = tf.summary.scalar("c_loss_x", c_loss_x)
    c_loss_z_sum = tf.summary.scalar("c_loss_z", c_loss_z)
    prior_loss_sum = tf.summary.scalar("prior_loss", prior_loss)
    img_sum = tf.summary.image("img", fake_x, max_outputs=10)

    theta_e = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder_z')
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_c_z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_mlp')
    theta_c_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    theta_prior = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='prior')

    counter_prior = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_prior = ly.optimize_loss(loss=prior_loss, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_prior, global_step=counter_prior,
                    summaries = 'gradient_norm')

    counter_c_z = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_z = ly.optimize_loss(loss=c_loss_z, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_c_z, global_step=counter_c_z,
                    summaries = 'gradient_norm')

    counter_c_x = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
    opt_c_x = ly.optimize_loss(loss=c_loss_x, learning_rate= None,
                    optimizer=tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9),
                    variables=theta_c_x, global_step=counter_c_x,
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

    return unlabeled_data, z, opt_c_z, opt_c_x, opt_e, opt_g, fake_x, c_loss_z, c_loss_x, e_loss, g_loss, unlabeled_z, opt_prior, prior_loss, sample_pi, tau


# In[9]:

def main():
    mus = np.concatenate([np.identity(10), np.zeros((10, z_dim-10))], 1)
    covs = np.stack([np.identity(z_dim) * args.sigma] * 10, 0)
    max_iter_step = 60000
    trainset = make_dataset(np.concatenate([trainx, testx],0), np.concatenate([trainy, testy],0))
    testset = make_dataset(testx, testy)
    with tf.device(device):
        unlabeled_data, z, opt_c_z, opt_c_x, opt_e, opt_g, fake_x, c_loss_z, c_loss_x, e_loss, g_loss, unlabeled_z, opt_prior, prior_loss, sample_pi, tau = build_graph()
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = False
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        tau0=1.0 # initial temperature
        np_temp=tau0
        ANNEAL_RATE=0.00003
        MIN_TEMP=0.5
        print("Training unsupervised wgan...")
        for i in range(max_iter_step):
            for j in range(Citers):
                if i % 100 == 99 and j == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, _, merged, loss_c_z_, loss_c_x_ = sess.run([opt_c_z, opt_c_x, merged_all, c_loss_z, c_loss_x], feed_dict={unlabeled_data: trainset.next_batch(batch_size)[0], tau:np_temp}, options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                else:
                    _, _ = sess.run([opt_c_z, opt_c_x], feed_dict={tau:np_temp,unlabeled_data: trainset.next_batch(batch_size)[0]})

            if i % 100 == 99:
                _, _, merged, loss_g_, loss_prior_ = sess.run([opt_g, opt_prior, merged_all, g_loss, prior_loss], feed_dict={unlabeled_data: trainset.next_batch(batch_size)[0], tau:np_temp}, options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'generator_and_prior_metadata {}'.format(i), i)

                _, merged, loss_e_ = sess.run([opt_e, merged_all, e_loss], feed_dict={unlabeled_data: trainset.next_batch(batch_size)[0], tau:np_temp}, options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'encoder_metadata {}'.format(i), i)
            else:
                _, _ = sess.run([opt_g, opt_prior], feed_dict={tau:np_temp,unlabeled_data: trainset.next_batch(batch_size)[0]})
                _ = sess.run([opt_e], feed_dict={tau:np_temp,unlabeled_data: trainset.next_batch(batch_size)[0]})
                
            if i % 100 == 99:
                # pi_ = sess.run(sample_pi, feed_dict={tau:np_temp})
                # print(pi_)
                print("Training ite %d, c_loss_z: %f, c_loss_x: %f, prior_loss: %f, e_loss: %f, g_loss: %f" % (i, loss_c_z_, loss_c_x_, loss_prior_, loss_e_, loss_g_))
                # batch_y = []
                # for j in range(10):
                #     for k in range(10):
                #         batch_y.append(j)
                # batch_z = np.concatenate([sample_z_given_y(gmm, y) for y in batch_y[:batch_size]], 0)
                # bx = sess.run(fake_x, feed_dict={z: batch_z})
                # batch_z = np.concatenate([sample_z_given_y(gmm, y) for y in batch_y[100-batch_size:]], 0)
                # bx1 = sess.run(fake_x, feed_dict={z: batch_z})
                # bx = np.concatenate([bx, bx1[2*batch_size-100:batch_size]], 0)
                # fig = plt.figure(image_dir + '.clustering-wgan')
                # grid_show(fig, (bx + 1) / 2, [32, 32, channel])
                # if not os.path.exists('./logs/{}/{}'.format(image_dir, args.logdir)):
                #     os.makedirs('./logs/{}/{}'.format(image_dir, args.logdir))
                # fig.savefig('./logs/{}/{}/{}.png'.format(image_dir, args.logdir, (i-99)/100))

            if i % 1000 == 999:
                np_temp=np.maximum(tau0*np.exp(-ANNEAL_RATE*i),MIN_TEMP)

                trainset.shuffle()
                true_zs = np.zeros((trainset._num_examples / batch_size * batch_size, z_dim))
                gts = np.zeros((trainset._num_examples / batch_size * batch_size))
                for j in range(trainset._num_examples / batch_size):
                    train_img, train_label = trainset.next_batch(batch_size)
                    bz = sess.run(unlabeled_z, feed_dict={unlabeled_data: train_img})
                    true_zs[j*batch_size:(j+1)*batch_size] = bz
                    gts[j*batch_size:(j+1)*batch_size] = train_label
                preds = cluster.KMeans(n_clusters=10, n_jobs=-1).fit_predict(true_zs)
                print("Training ite %d, acc: %f, nmi: %f" % (i, cluster_acc(preds, gts), cluster_nmi(preds, gts)))

                saver.save(sess, os.path.join(
                    ckpt_dir, "model.ckpt"), global_step=i)


main()
