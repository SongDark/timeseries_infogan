import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf 
from datamanager import datamanager
from utils import *
import os

class Generator(BasicBlock):
    def __init__(self, output_dim, name=None):
        super(Generator, self).__init__(None, name or "G")
        self.output_dim = output_dim
    
    def __call__(self, z, y=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = z.get_shape().as_list()[0]
            if y is not None:
                z = tf.concat([z,y], 1)

            net = tf.nn.relu(bn(dense(z, 1024, name='g_fc1'), is_training, name='g_bn1'))
            net = tf.nn.relu(bn(dense(net, 128*46, name='g_fc2'), is_training, name='g_bn2'))
            net = tf.reshape(net, [batch_size, 46, 1, 128])
            # [bz, 92, 1, 64]
            net = tf.nn.relu(bn(deconv2d(net, 64, 4, 1, 2, 1, padding='SAME', name='g_dc3'), is_training, name='g_bn3'))
            # [bz, 184, 1, 32]
            net = deconv2d(net, 1, 4, 2, 2, 2, padding='SAME', name='g_dc4')
            # [bz, 182, 1, output_dim]
            net = conv2d(net, self.output_dim, 3, 1, 1, 1, padding='VALID', name='g_c5')
        return net

class Discriminator(BasicBlock):
    def __init__(self, class_num=None, name=None):
        super(Discriminator, self).__init__(None, name or "D")
        self.class_num = class_num
    
    def __call__(self, x, y=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = x.get_shape().as_list()[0]
            if y is not None:
                ydim = y.get_shape().as_list()[-1]
                y = tf.reshape(y, [batch_size, 1, 1, ydim])
                x = conv_cond_concat(x, y)
            # [bz, 91, 1, 64]
            net = lrelu(conv2d(x, 32, 4, 1, 2, 1, padding="SAME", name='d_c1'), name='d_l1')
            # [bz, 46, 1, 128]
            net = lrelu(bn(conv2d(net, 64, 4, 1, 2, 1, padding="SAME", name='d_c2'), is_training, name='d_bn2'), name='d_l2')
            net = tf.reshape(net, [batch_size, -1])
            # [bz, 1024]
            net = lrelu(bn(dense(net, 1024, name='d_fc3'), is_training, name='d_bn3'), name='d_l3')
            # [bz, 1]
            yd = dense(net, 1, name='D_dense')
            if self.class_num:
                yc = dense(net, self.class_num, name='C_dense')
                return yd, net, yc 
            else:
                return yd, net

class Classifier(BasicBlock):
    def __init__(self, class_num, name=None):
        super(Classifier, self).__init__(None, name or 'C')
        self.class_num = class_num
    
    def __call__(self, x, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            net = lrelu(bn(dense(x, 64, name='c_fc1'), is_training, name='c_bn1'), name='c_l1')
            out_logit = dense(net, self.class_num, name='c_fc3')
            out = tf.nn.softmax(out_logit)
        return out_logit, out

# z = tf.ones(shape=(64, 100), dtype=tf.float32)
# G = Generator(3)
# D = Discriminator()
# C = Classifier(22)
# g = G(z)
# yd, net = D(g)
# out_logit, out = C(net)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(g).shape
#     print sess.run(yd).shape
#     print sess.run(net).shape
#     # print sess.run(yc).shape
#     print sess.run(out_logit).shape
#     print sess.run(out).shape

class infoGAN(BasicTrainFramework):
    def __init__(self, batch_size, version):
        super(infoGAN, self).__init__(batch_size, version or 'infoGAN')

        self.noise_dim = 100
        self.SUPERVISED = True

        self.data = datamanager("data/CharacterTrajectories/CharacterTrajectories.npz", 
                   train_ratio=1.0, fold_k=None, norm=None, expand_dim=3, seed=0)
        self.class_num = self.data.class_num

        # code
        self.len_code = 2

        self.Generator = Generator(output_dim=1, name='G')
        self.Discriminator = Discriminator(name='D')
        self.Classifier = Classifier(class_num=self.class_num + self.len_code, name='C')

        self.build_placeholder()
        self.build_gan()
        self.build_optimizer()
        self.build_summary()
        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.noise = tf.placeholder(shape=(self.batch_size, self.noise_dim), dtype=tf.float32)
        self.source = tf.placeholder(shape=(self.batch_size, 182, 2, 1), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(self.batch_size, self.class_num + self.len_code), dtype=tf.float32)

    def build_gan(self):
        self.G = self.Generator(self.noise, self.labels, is_training=True, reuse=False)
        self.G_test = self.Generator(self.noise, self.labels, is_training=False, reuse=True)
        self.logit_real, self.conv_real = self.Discriminator(self.source, is_training=True, reuse=False)
        self.logit_fake, self.conv_fake = self.Discriminator(self.G, is_training=True, reuse=True)

        self.logit_cls, self.softmax_cls = self.Classifier(self.conv_fake)
    
    def build_optimizer(self):
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
        
        # discrete code : categorical
        disc_code_est = self.softmax_cls[:, :self.class_num]
        disc_code_tg = self.labels[:, :self.class_num]
        self.q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_code_est, labels=disc_code_tg))
        # continuous code : gaussian
        cont_code_est = self.softmax_cls[:, self.class_num:]
        cont_code_tg = self.labels[:, self.class_num:]
        self.q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))
        self.Q_loss = self.q_disc_loss + self.q_cont_loss

        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss, var_list=self.Discriminator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(self.G_loss, var_list=self.Generator.vars)
        self.Q_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(self.Q_loss, var_list=self.Classifier.vars + self.Generator.vars + self.Discriminator.vars)

    def build_summary(self):
        D_sum = tf.summary.scalar("D_loss", self.D_loss)
        D_sum_real = tf.summary.scalar("D_loss_real", self.D_loss_real)
        D_sum_fake = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
        G_sum = tf.summary.scalar("G_loss", self.G_loss)
        Q1_sum = tf.summary.scalar("q_disc_loss", self.q_disc_loss)
        Q2_sum = tf.summary.scalar("q_cont_loss", self.q_cont_loss)
        Q_sum = tf.summary.scalar("Q_loss", self.Q_loss)
        self.summary = tf.summary.merge([D_sum, D_sum_real, D_sum_fake, G_sum, Q_sum, Q1_sum, Q2_sum])
    
    def sample(self, epoch):
        # test_codes = np.tile(np.linspace(-2,2,8), 8)[:,None]
        # test_codes = np.concatenate([labels, test_codes, np.zeros((64,1))], 1)

        '''random noise, random discrete code, fixed continuous code'''
        y = np.random.choice(self.class_num, self.batch_size)
        y_one_hot = one_hot_encode(y, self.class_num + self.len_code)
        z_sample = np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        G = self.sess.run(self.G_test, feed_dict={self.noise:z_sample, self.labels:y_one_hot})
        for i in range(8):
            for j in range(8):
                idx = i*8+j
                plt.subplot(8,8,idx+1)
                plt.plot(G[idx, :, 0, 0], G[idx, :, 1, 0])
                plt.xticks([])
                plt.yticks([])
        plt.savefig(os.path.join(self.fig_dir, "fake_all_classes_epoch{}.png".format(epoch)))
        plt.clf()

        '''specified condition, random noise'''
        n = 8
        si = np.random.choice(self.batch_size, n)
        for l in range(self.class_num):
            y = np.zeros(self.batch_size, dtype=np.int32) + l
            y_one_hot = one_hot_encode(y, self.class_num + self.len_code)
            G = self.sess.run(self.G_test, feed_dict={self.noise:z_sample, self.labels:y_one_hot})
            G = G[si, :, :, :]
            if l == 0:
                all_samples = G 
            else:
                all_samples = np.concatenate((all_samples, G), axis=0)
        for i in range(n):
            for j in range(n):
                idx = i*8+j
                plt.subplot(8,8,idx+1)
                plt.plot(all_samples[idx, :, 0, 0], all_samples[idx, :, 1, 0])
                plt.xticks([])
                plt.yticks([])
        plt.savefig(os.path.join(self.fig_dir, "fake_all_classes_style_by_style_epoch{}.png".format(epoch)))
        plt.clf()

        '''fixed noise'''
        z_fixed = np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        for label_idx in range(0, 20, 2):
            labels = one_hot_encode(np.array([label_idx] * self.batch_size), self.class_num)
            
            test_codes = np.zeros((8*8, 2))
            tmp = np.linspace(-0.99, 0.99, 8)
            for i in range(8):
                for j in range(8):
                    test_codes[i*8+j, :] = [tmp[i], tmp[j]]
            test_codes = np.concatenate([labels, test_codes], 1)

            feed_dict = {
                self.noise: z_fixed,
                self.labels: test_codes
            }
            G = self.sess.run(self.G_test, feed_dict=feed_dict)

            for i in range(8):
                for j in range(8):
                    idx = i*8+j
                    plt.subplot(8,8,idx+1)
                    plt.plot(G[idx, :, 0, 0], G[idx, :, 1, 0])
                    plt.xticks([])
                    plt.yticks([])
            plt.savefig(os.path.join(self.fig_dir, "fake_label{}_epoch{}.png".format(label_idx, epoch)))
            plt.clf()


    
    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            self.data.shuffle_train(seed=epoch)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data(self.batch_size, var_list=["data", "labels"])
                
                # [bz, 10]
                if self.SUPERVISED:
                    batch_labels = data["labels"] 
                else:
                    batch_labels = np.random.multinomial(1, self.class_num * [float(1.0/self.class_num)], size=[self.batch_size])
                # [bz, 12]
                batch_codes = np.concatenate((batch_labels, np.random.uniform(-1,1,size=(self.batch_size, self.len_code))), axis=1)

                feed_dict = {
                    self.source: data["data"],
                    self.labels: batch_codes,
                    self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
                }

                # train D
                self.sess.run(self.D_solver, feed_dict=feed_dict)

                # train G
                if (cnt-1) % 1 == 0:
                    self.sess.run(self.G_solver, feed_dict=feed_dict)
                
                # train Q 
                self.sess.run(self.Q_solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    d_loss, d_loss_r, d_loss_f, g_loss, q_loss, q1_loss, q2_loss, sum_str = self.sess.run([self.D_loss, self.D_loss_real, self.D_loss_fake, self.G_loss, self.Q_loss, self.q_disc_loss, self.q_cont_loss,self.summary], feed_dict=feed_dict)
                    print self.version + " epoch [%3d/%3d] iter [%3d/%3d] D=%.3f Dr=%.3f Df=%.3f G=%.3f Q=%.3f Q1=%.3f Q2=%.3f" % \
                        (epoch, epoches, idx, batches_per_epoch, d_loss, d_loss_r, d_loss_f, g_loss, q_loss, q1_loss, q2_loss)
                    self.writer.add_summary(sum_str, cnt)

            if epoch % 100 == 0:
                self.sample(epoch)
        # self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)

        
        
    
    
gan = infoGAN(64, 'infogan')
gan.train(501)