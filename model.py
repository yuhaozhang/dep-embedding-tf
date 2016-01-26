import tensorflow as tf
import numpy as np
import random, math
from prepare_data import *

class ModelOptions:
    def __init__(self):
        self.BATCH_SIZE = 256
        self.EMBED_SIZE = 100
        self.NUM_NEG_SAMPLE = 15

        self.TOTAL_OPOCHS = 6
        self.VALIDSET_SIZE = 10

        self.LR = 0.25

        self.LOG_STEP = 5000
        self.EVAL_STEP = 30000

class Model:
    def __init__(self, options, data):
        assert(isinstance(options, ModelOptions))
        assert(isinstance(data, Data))
        self.options = options
        self.data = data
        self.num_step = 0

        if self.data.isprepared == False:
            data.prepare_data()

        self.valid_examples = np.array(random.sample(np.arange(data.word_vocab_size), self.options.VALIDSET_SIZE))
        self.valid_examples[0] = self.data.word2idx['a']
        self.valid_examples[1] = self.data.word2idx['on']
        self.valid_examples[2] = self.data.word2idx['it']
        self.valid_examples[3] = self.data.word2idx['get']

    def get_data_nodes(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.options.BATCH_SIZE])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.options.BATCH_SIZE,1])
        self.validset = tf.constant(self.valid_examples, dtype=tf.int32)

    def build_graph(self):
        opt = self.options
        self.word_embeddings = tf.Variable(tf.random_uniform([self.data.word_vocab_size, opt.EMBED_SIZE], -1.0, 1.0))
        self.ctx_embedding_weights = tf.Variable(tf.truncated_normal([self.data.ctx_vocab_size, opt.EMBED_SIZE], 
            stddev=1.0/math.sqrt(opt.EMBED_SIZE)))
        self.ctx_embedding_biases = tf.Variable(tf.zeros([self.data.ctx_vocab_size]))

        embed = tf.nn.embedding_lookup(self.word_embeddings, self.train_inputs)

        self.loss = tf.reduce_mean(tf.nn.nce_loss(self.ctx_embedding_weights, self.ctx_embedding_biases, embed, self.train_labels,
            opt.NUM_NEG_SAMPLE, self.data.ctx_vocab_size))

    def build_eval_graph(self):
        norm = tf.sqrt(tf.reduce_mean(tf.square(self.word_embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.word_embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.word_embeddings, self.validset)
        self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)

    def run_training(self):
        opt = self.options
        graph = tf.Graph()
        with graph.as_default():
            self.get_data_nodes()
            self.build_graph()
            self.build_eval_graph()
            self.optimizer = tf.train.GradientDescentOptimizer(opt.LR).minimize(self.loss)

        print 'Start training...'
        print 'Estimated total steps needed: %d.' % (self.data.total_examples * opt.TOTAL_OPOCHS / opt.BATCH_SIZE)
        with tf.Session(graph=graph) as sess:
            tf.initialize_all_variables().run()
            average_loss = 0
            while self.data.epoch < opt.TOTAL_OPOCHS:
                batch_inputs, batch_labels = self.data.get_batch(opt.BATCH_SIZE)
                _, loss_val = sess.run([self.optimizer, self.loss], 
                    feed_dict={self.train_inputs:batch_inputs, self.train_labels:batch_labels})
                self.num_step += 1
                average_loss += loss_val

                if self.num_step % opt.LOG_STEP == 0:
                    average_loss = average_loss / opt.LOG_STEP
                    print "Average loss at step %d: %g" % (self.num_step, average_loss)
                    average_loss = 0

                if self.num_step % opt.EVAL_STEP == 0:
                    sim = self.similarity.eval()
                    top_k = 5
                    for i in range(opt.VALIDSET_SIZE):
                        nearest = (-sim[i,:]).argsort()[1:top_k+1]
                        nearest_words = [self.data.idx2word[j] for j in nearest]
                        log_str = "Nearest to %s: " % self.data.idx2word[self.valid_examples[i]]
                        print log_str + str(nearest_words)
            final_embeddings = self.normalized_embeddings.eval()

if __name__ == '__main__':
    data_opt = DataOptions()
    data_opt.CONLL_FILE = 'medium_data.conll'
    data_opt.compile()
    data = Data(data_opt)
    data.prepare_data()

    model_opt = ModelOptions()
    model = Model(model_opt, data)
    model.run_training()


