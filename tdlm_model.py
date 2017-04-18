import tensorflow as tf
import numpy as np
import math
import scipy.stats
from gensim import matutils
from tensorflow.python.ops import array_ops
if tf.__version__.split(".")[0] == "0":
    if int(tf.__version__.split(".")[1]) > 8:
        from tensorflow.python.ops.rnn_cell import _linear as linear
    else:
        from tensorflow.python.ops.rnn_cell import linear
else:
    print "TDLM supports tensorflow 0.8-0.12 only"
    raise SystemExit

#convolutional topic model
class TopicModel(object):
    def __init__(self, is_training, vocab_size, batch_size, num_steps, num_classes, config, reuse_conv_variables=None):
        self.conv_size = len(config.filter_sizes) * config.filter_number
        self.config = config #save config

        #placeholders
        self.y = tf.placeholder(tf.int32, [None, num_steps])
        self.tm_mask = tf.placeholder(tf.float32, [None, num_steps])
        self.doc = tf.placeholder(tf.int32, [None, config.doc_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.sup_mask = tf.placeholder(tf.float32, [None])
        self.tag = tf.placeholder(tf.int32, [None, config.tag_len])

        #variables
        with tf.variable_scope("tm_var", reuse=reuse_conv_variables):
            self.conv_word_embedding = tf.get_variable("conv_embedding", [vocab_size, config.word_embedding_size], \
                trainable=config.word_embedding_update, \
                initializer=tf.random_uniform_initializer(-0.5/config.word_embedding_size, \
                0.5/config.word_embedding_size))
            self.topic_output_embedding = tf.get_variable("topic_output_embedding", \
                [config.topic_number, config.topic_embedding_size])
            self.topic_input_embedding = tf.get_variable("topic_input_embedding", \
                [config.topic_number, self.conv_size+config.tag_embedding_size])
            self.tm_softmax_w = tf.get_variable("tm_softmax_w", [config.topic_embedding_size, vocab_size])
            if is_training and config.num_samples > 0:
                self.tm_softmax_w_t = tf.transpose(self.tm_softmax_w)
            self.tm_softmax_b = tf.get_variable("tm_softmax_b", [vocab_size], initializer=tf.constant_initializer())
            self.eye = tf.constant(np.eye(config.topic_number), dtype=tf.float32)
            if num_classes > 0:
                self.sup_softmax_w = tf.get_variable("sup_softmax_w", \
                    [config.topic_embedding_size+self.conv_size+config.tag_embedding_size, num_classes])
                self.sup_softmax_b = tf.get_variable("sup_softmax_b", [num_classes], \
                    initializer=tf.constant_initializer())
            if config.num_tags > 0:
                self.tag_embedding = tf.get_variable("tag_embedding", [config.num_tags, config.tag_embedding_size], \
                    initializer=tf.random_uniform_initializer(-0.001, 0.001))
                

        #get document embedding
        doc_inputs = tf.nn.embedding_lookup(self.conv_word_embedding, self.doc)
        if is_training and config.tm_keep_prob < 1.0:
            doc_inputs = tf.nn.dropout(doc_inputs, config.tm_keep_prob, seed=config.seed)
        doc_inputs = tf.expand_dims(doc_inputs, -1)

        #apply convolutional filters on the words
        pooled_outputs = []
        for i, filter_size in enumerate(config.filter_sizes):
            with tf.variable_scope("tm_var/conv-filter-%d" % filter_size, reuse=reuse_conv_variables):
                filter_w = tf.get_variable("filter_w", \
                    [filter_size, config.word_embedding_size, 1, config.filter_number], \
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                filter_b = tf.get_variable("filter_b", [config.filter_number], \
                    initializer=tf.constant_initializer())
                conv = tf.nn.conv2d(doc_inputs, filter_w, strides=[1,1,1,1], padding="VALID")
                if config.conv_activation == "identity":
                    conv_activated = tf.nn.bias_add(conv, filter_b)
                elif config.conv_activation == "relu":
                    conv_activated = tf.nn.relu(tf.nn.bias_add(conv, filter_b))

                #max pooling over time steps
                h = tf.nn.max_pool(conv_activated, \
                    ksize=[1,(config.doc_len-filter_size+1),1,1], strides=[1,1,1,1], padding="VALID")
                pooled_outputs.append(h)

        #concat the pooled features
        conv_pooled = tf.concat(3, pooled_outputs)
        conv_pooled = tf.reshape(conv_pooled, [-1, self.conv_size])

        #if there are tags, compute sum embedding and concat it with conv_pooled
        if config.num_tags > 0:
            tag_emb = tf.nn.embedding_lookup(self.tag_embedding, self.tag)
            tag_mask = tf.expand_dims(tf.cast(tf.greater(self.tag, 0), tf.float32),2)
            tag_emb_m = tf.reduce_sum(tag_emb*tag_mask, 1)
            conv_pooled = tf.concat(1, [conv_pooled, tag_emb_m])

        #get the softmax attention weights and compute mean topic vector
        self.attention = tf.nn.softmax(tf.reduce_sum(tf.mul(tf.expand_dims(self.topic_input_embedding, 0), \
            tf.expand_dims(conv_pooled, 1)), 2))
        self.mean_topic = tf.reduce_sum(tf.mul(tf.expand_dims(self.attention, 2), \
            tf.expand_dims(self.topic_output_embedding, 0)),1)
        if is_training and config.tm_keep_prob < 1.0:
            self.mean_topic_dropped = tf.nn.dropout(self.mean_topic, config.tm_keep_prob, seed=config.seed)
        else:
            self.mean_topic_dropped = self.mean_topic

        #reshape mean_topic from [batch_size,config.topic_embedding_size] to 
        #[batch_size*sent_len,config.topic_embedding_size]
        self.conv_hidden = tf.reshape(tf.tile(self.mean_topic_dropped, [1, num_steps]), \
            [batch_size*num_steps, config.topic_embedding_size])

        #compute masked/weighted crossent and mean topic model loss
        if is_training and config.num_samples > 0:
            tm_crossent = tf.nn.sampled_softmax_loss(self.tm_softmax_w_t, self.tm_softmax_b, self.conv_hidden, \
                tf.reshape(self.y, [-1,1]), config.num_samples, vocab_size)
        else:
            self.tm_logits = tf.matmul(self.conv_hidden, self.tm_softmax_w) + self.tm_softmax_b
            tm_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(self.tm_logits, tf.reshape(self.y, [-1]))
        tm_crossent_m = tm_crossent * tf.reshape(self.tm_mask, [-1])
        self.tm_cost = tf.reduce_sum(tm_crossent_m) / batch_size

        #compute supervised classification loss
        if num_classes > 0:
            sup_hidden = tf.concat(1, [conv_pooled, self.mean_topic])
            sup_logits = tf.matmul(sup_hidden, self.sup_softmax_w) + self.sup_softmax_b
            self.sup_probs = tf.nn.softmax(sup_logits)
            sup_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(sup_logits, self.label)
            sup_crossent_m = sup_crossent * self.sup_mask
            self.sup_cost = tf.reduce_sum(sup_crossent_m) / batch_size

        if not is_training:
            return

        #topic uniqueness loss
        topicnorm = self.topic_output_embedding / tf.sqrt(tf.reduce_sum(tf.square(self.topic_output_embedding),1, \
            keep_dims=True))
        uniqueness = tf.reduce_max(tf.square(tf.matmul(topicnorm, tf.transpose(topicnorm)) - self.eye))
        self.tm_cost += config.alpha * uniqueness

        #entropy = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.maximum(self.attention, 1e-3))*(self.attention),1))
        #self.tm_cost += config.alpha * entropy

        #run optimiser and backpropagate gradients for tm loss
        self.tm_train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.tm_cost)

        if num_classes > 0:
            #run optimiser and backpropagate gradients for supervision loss
            self.sup_train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.sup_cost)

    #get top-N highest probability words for each topic
    def get_topics(self, sess, topn):
        topics = []
        entropy = []
        tw_dist = sess.run(tf.nn.softmax(tf.matmul(self.topic_output_embedding, self.tm_softmax_w) + self.tm_softmax_b))
        for ti in xrange(self.config.topic_number):
            best = matutils.argsort(tw_dist[ti], topn=topn, reverse=True)
            topics.append(best)
            entropy.append(scipy.stats.entropy(tw_dist[ti]))

        return topics, entropy

    #get top topics and words given a doc
    def get_topics_on_doc(self, sess, doc, tag, topn):
        tw_dist, logits = sess.run([self.attention, self.tm_logits], {self.doc: doc, self.tag: tag})
        probs = sess.run(tf.nn.softmax(logits))[0]
        best_words = matutils.argsort(probs, topn=topn, reverse=True)
        best_words = [ (item, probs[item]) for item in best_words ] #attach word probability
        best_topics = matutils.argsort(tw_dist[0], topn=topn, reverse=True)
        best_topics = [ (item, tw_dist[0][item]) for item in best_topics ] #attach topic probability

        return best_topics, best_words

#convolutional topic model + lstm language model
class LanguageModel(TopicModel):
    def __init__(self, is_training, vocab_size, batch_size, num_steps, config, reuse_conv_variables=None):
        if config.topic_number > 0:
            TopicModel.__init__(self, is_training, vocab_size, batch_size, num_steps, 0, config, reuse_conv_variables)
        else:
            self.y = tf.placeholder(tf.int32, [None, num_steps])
            self.config = config

        #placeholders
        self.x = tf.placeholder(tf.int32, [None, num_steps])
        self.lm_mask = tf.placeholder(tf.float32, [None, num_steps])

        #variables
        self.lstm_word_embedding = tf.get_variable("lstm_embedding", [vocab_size, config.word_embedding_size], \
            trainable=config.word_embedding_update, \
            initializer=tf.random_uniform_initializer(-0.5/config.word_embedding_size, 0.5/config.word_embedding_size))
        self.lm_softmax_w = tf.get_variable("lm_softmax_w", [config.rnn_hidden_size, vocab_size])
        if is_training and config.num_samples > 0:
            self.lm_softmax_w_t = tf.transpose(self.lm_softmax_w)
        self.lm_softmax_b = tf.get_variable("lm_softmax_b", [vocab_size], initializer=tf.constant_initializer())
        if config.topic_number > 0:
            self.gate_w = tf.get_variable("gate_w", [config.topic_embedding_size, config.rnn_hidden_size])
            self.gate_u = tf.get_variable("gate_u", [config.rnn_hidden_size, config.rnn_hidden_size])
            self.gate_b = tf.get_variable("gate_b", [config.rnn_hidden_size], initializer=tf.constant_initializer())

        #define lstm cells
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.rnn_hidden_size, forget_bias=1.0)
        if is_training and config.lm_keep_prob < 1.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.lm_keep_prob, seed=config.seed)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.rnn_layer_size)

        #set initial state to all zeros
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        #embedding lookup
        inputs = tf.nn.embedding_lookup(self.lstm_word_embedding, self.x)
        if is_training and config.lm_keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, config.lm_keep_prob, seed=config.seed)

        #transform input from [batch_size,sent_len,emb_size] to [sent_len,batch_size,emb_size ]
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]

        #run rnn and get outputs (hidden layer)
        outputs, self.state = tf.nn.rnn(self.cell, inputs, initial_state=self.initial_state)

        #reshape output into [sent_len,batch_size,hidden_size] and then into [batch_size*sent_len,hidden_size]
        lstm_hidden = tf.reshape(tf.concat(1, outputs), [-1, config.rnn_hidden_size])

        if config.topic_number > 0:
            #combine topic and language model hidden with a gating unit
            z, r = array_ops.split(1, 2, linear([self.conv_hidden, lstm_hidden], \
                2 * config.rnn_hidden_size, True, 1.0))
            z, r = tf.sigmoid(z), tf.sigmoid(r)
            c = tf.tanh(tf.matmul(self.conv_hidden, self.gate_w) + tf.matmul((r * lstm_hidden), self.gate_u) + \
                self.gate_b)
            hidden = (1-z)*lstm_hidden + z*c
            
            #save z
            self.tm_weights = tf.reshape(tf.reduce_mean(z, 1), [-1, num_steps])
        else:
            hidden = lstm_hidden

        #compute masked/weighted crossent and mean language model loss
        if is_training and config.num_samples > 0:
            lm_crossent = tf.nn.sampled_softmax_loss(self.lm_softmax_w_t, self.lm_softmax_b, hidden, \
                tf.reshape(self.y, [-1,1]), config.num_samples, vocab_size)
        else:
            lm_logits = tf.matmul(hidden, self.lm_softmax_w) + self.lm_softmax_b
            lm_crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(lm_logits, tf.reshape(self.y, [-1]))
        lm_crossent_m = lm_crossent * tf.reshape(self.lm_mask, [-1])
        self.lm_cost = tf.reduce_sum(lm_crossent_m) / batch_size

        #compute probs if in testing mode
        if not is_training:
            self.probs = tf.nn.softmax(lm_logits)
            return

        #run optimiser and backpropagate (clipped) gradients for lm loss
        lm_tvars = tf.trainable_variables()
        lm_grads, _ = tf.clip_by_global_norm(tf.gradients(self.lm_cost, lm_tvars), config.max_grad_norm)
        self.lm_train_op = tf.train.AdamOptimizer(config.learning_rate).apply_gradients(zip(lm_grads, lm_tvars))

    #sample a word given probability distribution (with option to normalise the distribution with temperature)
    #temperature = 0 means argmax
    def sample(self, probs, temperature):
        if temperature == 0:
            return np.argmax(probs)

        probs = probs.astype(np.float64) #convert to float64 for higher precision
        probs = np.log(probs) / temperature
        probs = np.exp(probs) / math.fsum(np.exp(probs))
        return np.argmax(np.random.multinomial(1, probs, 1))

    #generate a sentence given conv_hidden
    def generate(self, sess, conv_hidden, start_word_id, temperature, max_length, stop_word_id):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        x = [[start_word_id]]
        sent = [start_word_id]

        for _ in xrange(max_length):
            if type(conv_hidden) is np.ndarray:
            #if conv_hidden != None:
                probs, state = sess.run([self.probs, self.state], \
                    {self.x: x, self.initial_state: state, self.conv_hidden: conv_hidden})
            else:
                probs, state = sess.run([self.probs, self.state], \
                    {self.x: x, self.initial_state: state})
            sent.append(self.sample(probs[0], temperature))
            if sent[-1] == stop_word_id:
                break
            x = [[ sent[-1] ]]

        return sent

    #generate a sequence of words, given a topic
    def generate_on_topic(self, sess, topic_id, start_word_id, temperature=1.0, max_length=30, stop_word_id=None): 
        if topic_id != -1:
            topic_emb = sess.run(tf.expand_dims(tf.nn.embedding_lookup(self.topic_output_embedding, topic_id), 0))
        else:
            topic_emb = None
        return self.generate(sess, topic_emb, start_word_id, temperature, max_length, stop_word_id)

    #generate a sequence of words, given a document
    def generate_on_doc(self, sess, doc, tag, start_word_id, temperature=1.0, max_length=30, stop_word_id=None): 
        c = sess.run(self.conv_hidden, {self.doc: doc, self.tag: tag})
        return self.generate(sess, c, start_word_id, temperature, max_length, stop_word_id)
