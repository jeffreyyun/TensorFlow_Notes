import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from process_data import process_data


VOCAB_SIZE = 9999
BATCH_SIZE = 64
EMBED_SIZE = 64
SKIP_WINDOW = 2
NUM_SAMPLED = 64 # num of noise samples (per true sample)
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
WEIGHTS_FLD = 'processed/'
REPORT_STEP = 2000

# ESTABLISH MODEL
class SkipGramModel:
    """ builds the graph for word2vec model """
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        # number of total steps
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    
    def _create_placeholders(self):
        # Input/Output
        with tf.name_scope('data'):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')
    
    def _create_embedding(self):
        # Weight --> here, the Embedding Matrix
        with tf.name_scope('embed'):
            self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),
                                            name='embed_matrix')
    
    def _create_loss(self):
        with tf.name_scope('loss'):
            # Inference
            # tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
            # Look up the corresponding rows of center_words in the embedding matrix
            embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')
            # Loss function
            # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true=1, sampled_values=None, remove_accidental_hits=False, partition_strategy='mod', name='nce_loss')
            nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                         stddev=1.0 / self.embed_size ** 0.5), name='nce_weight')
            nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                                biases=nce_bias,
                                                labels=self.target_words,
                                                inputs=embed,
                                                num_sampled=self.num_sampled,
                                                num_classes=self.vocab_size), name='loss')
    
    def _create_optimizer(self):
        # Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.histogram("historgram loss", self.loss)
            # merge all summaries in one op to simplify -- only need to run one
            self.summary_op = tf.summary.merge_all()
            
    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

# TRAIN MODEL
def train_model(model, batch_gen, num_train_steps, weights_fld):
    saver = tf.train.Saver()
    
    initial_step = 0

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    # Saves a checkpoint!
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    total_loss = 0.0
    writer = tf.summary.FileWriter('improved_graph/lr' + str(LEARNING_RATE), sess.graph)
    initial_step = model.global_step.eval()
    for index in range(initial_step, initial_step + num_train_steps):
        # next batch
        centers, targets = batch_gen.next()
        feed_dict = {model.center_words: centers, target_words: targets}
        loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
        writer.add_summary(summary, global_step=index)
        total_loss += loss_batch
        if (index + 1) % REPORT_STEP == 0:
            print('Average loss at step {}: {:S.1f}'.format(index + 1, average_loss / (index + 1)))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/skip-gram', index)
    
    # visualize the embeddings
    final_embed_matrix = sess.run(model.embed_matrix)
    embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
    sess.run(embedding_var.initializer)
    
    # establish config file
    config = projector.ProjectorConfig()
    summary_writer = tf.summary.FileWriter('processed')
    
    # add embeddings to config
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    
    # links to metadata
    embedding.metadata_path = 'processed/vocab_1000.tsv'
    
    # saves config
    projector.visualize_embeddings(summary_writer, config)
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, 'processed/model3.ckpt', 1)
            
            
def main():
    model = SkipGramModel(VOCAB_SIZE, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)
    
if __name__ == '__main__':
    main()