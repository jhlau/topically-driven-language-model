"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Date:           Jul 16
"""

import argparse
import sys
import codecs
import random
import time
import os
import math
import pickle
import operator
import tensorflow as tf
import numpy as np
import gensim.models as g
import tdlm_config as cf
from util import pad, get_batch, get_batch_doc, gen_vocab, gen_data, print_corpus_stats, init_embedding, gen_tag
from tdlm_model import TopicModel as TM
from tdlm_model import LanguageModel as LM
from collections import defaultdict

#parser arguments
desc = "trains neural topic language model on a document collection (experiment settings defined in cf.py)"
parser = argparse.ArgumentParser(description=desc)
args = parser.parse_args()

#globals
vocabxid = {}
idxvocab = []

#constants
pad_symbol = "<pad>"
start_symbol = "<go>"
end_symbol = "<eos>"
unk_symbol = "<unk>"
dummy_symbols = [pad_symbol, start_symbol, end_symbol, unk_symbol]


###########
#functions#
###########

def fetch_batch_and_train(sents, docs, tags, model, seq_len, i, p1, p2):
    (tm_costs, tm_words, lm_costs, lm_words) = p1
    (m_tm_cost, m_tm_train, m_lm_cost, m_lm_train) = p2
    x, y, m, d, t = get_batch(sents, docs, tags, i, cf.doc_len, seq_len, cf.tag_len, cf.batch_size, 0, \
        (True if isinstance(model, LM) else False))

    if isinstance(model, LM):
        if cf.topic_number > 0:
            tm_cost, _, lm_cost, _ = sess.run([m_tm_cost, m_tm_train, m_lm_cost, m_lm_train], \
                {model.x: x, model.y: y, model.lm_mask: m, model.doc: d, model.tag: t})
        else:
            #pure lstm
            tm_cost, _, lm_cost, _ = sess.run([m_tm_cost, m_tm_train, m_lm_cost, m_lm_train], \
                {model.x: x, model.y: y, model.lm_mask: m})
    else:
        tm_cost, _, lm_cost, _ = sess.run([m_tm_cost, m_tm_train, m_lm_cost, m_lm_train], \
            {model.y: y, model.tm_mask: m, model.doc: d, model.tag: t})

    if tm_cost != None:
        tm_costs += tm_cost * cf.batch_size #keep track of full batch loss (not per example batch loss)
        tm_words += np.sum(m)
    if lm_cost != None:
        lm_costs += lm_cost * cf.batch_size
        lm_words += np.sum(m)

    return tm_costs, tm_words, lm_costs, lm_words

def run_epoch(sents, docs, labels, tags, models, is_training):

    ####unsupervised topic and language model training####

    #generate the batches
    tm_num_batches, lm_num_batches = int(math.ceil(float(len(sents[0]))/cf.batch_size)), \
        int(math.ceil(float(len(sents[1]))/cf.batch_size))
    batch_ids = list([ (item, 0) for item in range(tm_num_batches) ] + [ (item, 1) for item in range(lm_num_batches) ])
    seq_lens = (cf.tm_sent_len, cf.lm_sent_len)
    #shuffle batches and sentences
    random.shuffle(batch_ids)
    random.shuffle(sents[0])
    random.shuffle(sents[1])

    #set training and cost ops for topic and language model training
    tm_cost_ops = (tf.no_op(), tf.no_op(), tf.no_op(), tf.no_op())
    lm_cost_ops = (tf.no_op(), tf.no_op(), tf.no_op(), tf.no_op())
    if models[0] != None:
        tm_cost_ops = (models[0].tm_cost, (models[0].tm_train_op if is_training else tf.no_op()), tf.no_op(), tf.no_op())
    if models[1] != None:
        lm_cost_ops = (tf.no_op(), tf.no_op(), models[1].lm_cost, (models[1].lm_train_op if is_training else tf.no_op()))
    cost_ops = (tm_cost_ops, lm_cost_ops)

    start_time = time.time()
    lm_costs, tm_costs, lm_words, tm_words = 0.0, 0.0, 0.0, 0.0
    for bi, (b, model_id) in enumerate(batch_ids):
        tm_costs, tm_words, lm_costs, lm_words = fetch_batch_and_train(sents[model_id], docs[model_id], tags, \
            models[model_id], seq_lens[model_id], b, (tm_costs, tm_words, lm_costs, lm_words), cost_ops[model_id])

        #print progress
        output_string = "%d/%d: tm ppl = %.3f; lm ppl = %.3f; word/sec = %.1f" % \
            (bi+1, len(batch_ids), np.exp(tm_costs/max(tm_words, 1.0)), np.exp(lm_costs/max(lm_words, 1.0)),  \
            float(tm_words + lm_words)/(time.time()-start_time))
        print_progress(bi, len(batch_ids), is_training, output_string)

    ####supervised classification training####

    if labels != None:
        #randomise the batches
        batch_ids = range(int(math.ceil(float(len(docs[0]))/cf.batch_size)))
        random.shuffle(batch_ids)

        start_time = time.time()
        costs, accs = 0.0, []
        for bi, b in enumerate(batch_ids):
            d, y, m, t, num_docs = get_batch_doc(docs[0], labels, tags, b, cf.doc_len, cf.tag_len, cf.batch_size, 0)
            cost, prob, _ = sess.run([models[0].sup_cost, models[0].sup_probs, \
                (models[0].sup_train_op if is_training else tf.no_op())], \
                {models[0].doc:d, models[0].label:y, models[0].sup_mask: m, models[0].tag: t})
            costs += cost * cf.batch_size #keep track of full cost
            pred = np.argmax(prob, axis=1)
            accs.extend(pred[:num_docs] == y[:num_docs])

            #print progress
            output_string = "%d/%d: sup loss = %.3f; sup acc = %.3f; doc/sec = %.1f" % \
                (bi+1, len(batch_ids), costs/((bi+1)*cf.batch_size), np.mean(accs), \
                (bi+1)*cf.batch_size/(time.time()-start_time))
            print_progress(bi, len(batch_ids), is_training, output_string)
    else:
        accs = None
            
    return -np.mean(accs) if accs != None else np.exp(lm_costs/max(lm_words, 1.0))

def print_progress(bi, batch_total, is_training, output_string):
    if (((bi % 10) == 0) and cf.verbose) or (bi == batch_total-1):
        if is_training:
            sys.stdout.write("TRAIN ")
        else:
            sys.stdout.write("VALID ")
        sys.stdout.write(output_string)
        if bi == (batch_total-1):
            sys.stdout.write("\n")
        else:
            sys.stdout.write("\r")
        sys.stdout.flush()


######
#main#
######
#set the seeds
random.seed(cf.seed)
np.random.seed(cf.seed)

#set topic vector size and load word embedding model if given
if cf.word_embedding_model:
    print("Loading word embedding model...")
    mword = g.Word2Vec.load(cf.word_embedding_model)
    cf.word_embedding_size = mword.vector_size

#first pass to collect vocabulary information
print("First pass on train corpus to collect vocabulary stats...")
idxvocab, vocabxid, tm_ignore = gen_vocab(dummy_symbols, cf.train_corpus, cf.stopwords, cf.vocab_minfreq, \
    cf.vocab_maxfreq, cf.verbose)

#second pass to collect train/valid data for topic and language model
print("Processing train corpus to collect sentence and document data...")
train_sents, train_docs, train_docids, train_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.train_corpus, \
    cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)
print("Processing valid corpus to collect sentence and document data...")
valid_sents, valid_docs, valid_docids, valid_stats = gen_data(vocabxid, dummy_symbols, tm_ignore, cf.valid_corpus, \
    cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)

#labels given for documents
train_labels, valid_labels, num_classes = None, None, 0
if hasattr(cf, "train_target") and hasattr(cf, "valid_target"):
    train_labels = [ int(item) for item in open(cf.train_target).readlines() ]
    valid_labels = [ int(item) for item in open(cf.valid_target).readlines() ]
    num_classes = max(set(train_labels)) + 1
cf.num_classes = num_classes

#tags given for documents
train_tags, valid_tags, tagxid, tag_len = None, None, {} , 0
if hasattr(cf, "train_tag") and hasattr(cf, "valid_tag"):
    tagxid = gen_tag(cf.train_tag, pad_symbol)
    train_tags = [ [ tagxid[t] for t in line.strip().split("\t") if t in tagxid ] \
        for line in codecs.open(cf.train_tag, "r", "utf-8").readlines() ]
    valid_tags = [ [ tagxid[t] for t in line.strip().split("\t") if t in tagxid ] \
        for line in codecs.open(cf.valid_tag, "r", "utf-8").readlines() ]
    tag_len = max([ len(item) for item in train_tags ])
cf.num_tags = len(tagxid)
cf.tag_len = tag_len
if train_tags == None:
    cf.tag_embedding_size = 0

#delete lm data if it's not used
if cf.rnn_hidden_size == 0:
    train_sents = (train_sents[0], [])
    valid_sents = (valid_sents[0], [])

#delete tm data if it's not used
if cf.topic_number == 0:
    train_sents = ([], train_sents[1])
    valid_sents = ([], valid_sents[1])

#print some statistics of the data
print("Vocab size =", len(idxvocab))
if cf.num_classes > 0:
    print("Class size (supervised) =", cf.num_classes)
if cf.num_tags > 0:
    print("Tag size =", cf.num_tags - 1)
print_corpus_stats("Train corpus", train_sents, train_docs, train_stats)
print_corpus_stats("Valid corpus", valid_sents, valid_docs, valid_stats)

#train model
with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(cf.seed)
    initializer = tf.contrib.layers.xavier_initializer(seed=cf.seed)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        tm_train = TM(is_training=True, vocab_size=len(idxvocab), batch_size=cf.batch_size, \
            num_steps=cf.tm_sent_len, num_classes=num_classes, config=cf) if cf.topic_number > 0 else None
        lm_train = LM(is_training=True, vocab_size=len(idxvocab), batch_size=cf.batch_size, \
            num_steps=cf.lm_sent_len, config=cf, reuse_conv_variables=True) \
            if cf.rnn_hidden_size > 0  else None
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        tm_valid = TM(is_training=False, vocab_size=len(idxvocab), batch_size=cf.batch_size, \
            num_steps=cf.tm_sent_len, num_classes=num_classes, config=cf) if cf.topic_number > 0 else None
        lm_valid = LM(is_training=False, vocab_size=len(idxvocab), batch_size=cf.batch_size, \
            num_steps=cf.lm_sent_len, config=cf) if cf.rnn_hidden_size > 0 else None

    tf.initialize_all_variables().run()

    #initialise word embedding
    if cf.word_embedding_model:
        word_emb = init_embedding(mword, idxvocab)
        if cf.rnn_hidden_size > 0:
            sess.run(lm_train.lstm_word_embedding.assign(word_emb))
        if cf.topic_number > 0:
            sess.run(tm_train.conv_word_embedding.assign(word_emb))

    #save model every epoch
    if cf.save_model:
        if not os.path.exists(os.path.join(cf.output_dir, cf.output_prefix)):
            os.makedirs(os.path.join(cf.output_dir, cf.output_prefix))
        #create saver object to save model
        saver = tf.train.Saver()

    #train model
    prev_ppl = None
    for i in range(cf.epoch_size):
        print("\nEpoch =", i)
        #run a train epoch
        run_epoch(train_sents, train_docs, train_labels, train_tags, (tm_train, lm_train), True)
        #run a valid epoch
        curr_ppl = run_epoch(valid_sents, valid_docs, valid_labels, valid_tags, (tm_valid, lm_valid), False)
    
        if cf.save_model:
            if (i < 5) or (prev_ppl == None) or (curr_ppl < prev_ppl):
                saver.save(sess, os.path.join(cf.output_dir, cf.output_prefix, "model.ckpt"))
                prev_ppl = curr_ppl
            else:
                saver.restore(sess, os.path.join(cf.output_dir, cf.output_prefix, "model.ckpt"))
                print("\tNew valid performance > prev valid performance: restoring previous parameters...")

    #print top-N words from topics
    if cf.topic_number > 0:
        print("\nTopics\n======")
        topics, entropy = tm_train.get_topics(sess, topn=20)
        for ti, t in enumerate(topics):
            print("Topic", ti, "[", ("%.2f" % entropy[ti]), "] :", " ".join([ idxvocab[item] for item in t ]))

    #generate some random sentences
    if cf.rnn_hidden_size > 0:
        print("\nRandom Generated Sentences\n==========================")
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mgen = LM(is_training=False, vocab_size=len(idxvocab), batch_size=1, num_steps=1, config=cf, \
                reuse_conv_variables=True)
        for temp in [1.0, 0.75, 0.5]:
            print("\nTemperature =", temp)
            for _ in range(10):
                #select a random topic
                if cf.topic_number > 0:
                    topic = random.randint(0, cf.topic_number-1)
                    print("\tTopic", topic, ":",)
                else:
                    topic = -1
                    print("\t",)

                s = mgen.generate_on_topic(sess, topic, vocabxid[start_symbol], temp, cf.lm_sent_len+10, \
                    vocabxid[end_symbol])
                s = [ idxvocab[item] for item in s ]
                print(" ".join(s))

    #save model vocab and configurations
    if cf.save_model:
        #vocabulary information
        pickle.dump((idxvocab, tm_ignore, dummy_symbols), \
            open(os.path.join(cf.output_dir, cf.output_prefix, "vocab.pickle"), "wb"))

        #tag information
        if len(tagxid) > 0:
            pickle.dump(tagxid, open(os.path.join(cf.output_dir, cf.output_prefix, "tag.pickle"), "wb"))

        #create a dictionary object for config
        cf_dict = {}
        for k,v in vars(cf).items():
            if not k.startswith("__"):
                cf_dict[k] = v
        pickle.dump(cf_dict, open(os.path.join(cf.output_dir, cf.output_prefix, "config.pickle"), "wb"))
