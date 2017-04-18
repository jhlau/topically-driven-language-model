"""
Stdin:          N/A
Stdout:         N/A
Author:         Jey Han Lau
Date:           Oct 16
"""

import argparse
import sys
import os
import cPickle
import math
import codecs
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tdlm_model import TopicModel as TM
from tdlm_model import LanguageModel as LM
from util import pad, get_batch, get_batch_doc, gen_data, print_corpus_stats
from gensim import matutils

#parser arguments
desc = "Given a trained TDLM model, perform various test inferences"
parser = argparse.ArgumentParser(description=desc)

###################
#optional argument#
###################
parser.add_argument("-m", "--model_dir", required=True, help="directory of the saved model")
parser.add_argument("-d", "--input_doc", help="input file containing the test documents")
parser.add_argument("-l", "--input_label", help="input file containing the test labels")
parser.add_argument("-t", "--input_tag", help="input file containing the test tags")
parser.add_argument("--print_perplexity", help="print topic and language model perplexity of the input test documents", \
    action="store_true")
parser.add_argument("--print_acc", help="print supervised classification accuracy", action="store_true")
parser.add_argument("--output_topic", help="output file to save the topics (prints top-N words of each topic)")
parser.add_argument("--output_topic_dist", \
    help="output file to save the topic distribution of input docs (npy format)")
parser.add_argument("--output_tag_embedding", \
    help="output tag embeddings to file (npy format)")
parser.add_argument("--gen_sent_on_topic", help="generate sentences conditioned on topics")
parser.add_argument("--gen_sent_on_doc", help="generate sentences conditioned on input test documents")

args = parser.parse_args()

#parameters
topn=10 #number of top-N words to print for each topic
gen_temps = [1.0, 0.75] #temperatures for generation
gen_num = 3 #number of generated sentences
debug = False

###########
#functions#
###########

def compute_dt_dist(docs, labels, tags, model, max_len, batch_size, pad_id, idxvocab, output_file):
    #generate batches
    num_batches = int(math.ceil(float(len(docs)) / batch_size))
    dt_dist = []
    t = []
    combined = []
    docid = 0
    for i in xrange(num_batches):
        x, _, _, t, s = get_batch_doc(docs, labels, tags, i, max_len, cf.tag_len, batch_size, pad_id)
        attention, mean_topic = sess.run([model.attention, model.mean_topic], {model.doc: x, model.tag: t})
        dt_dist.extend(attention[:s])

        if debug:
            for si in xrange(s):
                d = x[si]
                print "\n\nDoc", docid, "=", " ".join([idxvocab[item] for item in d if (item != pad_id)])
                sorted_dist = matutils.argsort(attention[si], reverse=True)
                for ti in sorted_dist:
                    print "Topic", ti, "=", attention[si][ti]
                docid += 1

    np.save(open(output_file, "w"), dt_dist)

def run_epoch(sents, docs, tags, (tm, lm), pad_id, cf, idxvocab):
    #generate the batches
    tm_num_batches, lm_num_batches = int(math.ceil(float(len(sents[0]))/cf.batch_size)), \
        int(math.ceil(float(len(sents[1]))/cf.batch_size))

    #run an epoch to compute tm and lm perplexities
    if tm != None:
        tm_costs, tm_words = 0.0, 0.0
        for bi in xrange(tm_num_batches):
            _, y, m, d, t = get_batch(sents[0], docs[0], tags, bi, cf.doc_len, cf.tm_sent_len, cf.tag_len, cf.batch_size, \
                pad_id, False)
            tm_cost = sess.run(tm.tm_cost, {tm.y: y, tm.tm_mask: m, tm.doc: d, tm.tag: t})
            tm_costs += tm_cost * cf.batch_size
            tm_words += np.sum(m)
        print "\ntest topic model perplexity = %.3f" % (np.exp(tm_costs/tm_words))

    if lm != None:
        lm_costs, lm_words = 0.0, 0.0
        for bi in xrange(lm_num_batches):
            x, y, m, d, t = get_batch(sents[1], docs[1], tags, bi, cf.doc_len, cf.lm_sent_len, cf.tag_len, cf.batch_size, \
                pad_id, True)
            lm_cost, tw = sess.run([lm.lm_cost, lm.tm_weights], {lm.x: x, lm.y: y, lm.lm_mask: m, lm.doc: d, lm.tag: t})
            lm_costs += lm_cost * cf.batch_size
            lm_words += np.sum(m)

        print "test language model perplexity = %.3f" % (np.exp(lm_costs/lm_words))

def run_epoch_doc(docs, labels, tags, tm, pad_id, cf):
    batches = int(math.ceil(float(len(docs))/cf.batch_size))
    accs = []
    for b in xrange(batches):
        d, y, m, t, num_docs = get_batch_doc(docs, labels, tags, b, cf.doc_len, cf.tag_len, cf.batch_size, pad_id)
        prob = sess.run(tm.sup_probs, {tm.doc:d, tm.label:y, tm.sup_mask: m, tm.tag: t})
        pred = np.argmax(prob, axis=1)
        accs.extend(pred[:num_docs] == y[:num_docs])

    print "\ntest classification accuracy = %.3f" % np.mean(accs)

def gen_sent_on_topic(idxvocab, vocabxid, start_symbol, end_symbol, cf):
    output = codecs.open(args.gen_sent_on_topic, "w", "utf-8")
    topics, entropy = tm.get_topics(sess, topn=topn)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mgen = LM(is_training=False, vocab_size=len(idxvocab), batch_size=1, num_steps=1, config=cf, \
            reuse_conv_variables=True)

    for t in range(cf.topic_number):
        output.write("\n" + "="*100 + "\n")
        output.write("Topic " +  str(t) + ":\n")
        output.write(" ".join([ idxvocab[item] for item in topics[t] ]) + "\n\n")

        output.write("\nSentence generation (greedy; argmax):" + "\n")
        s = mgen.generate_on_topic(sess, t, vocabxid[start_symbol], 0, cf.lm_sent_len+10, vocabxid[end_symbol])
        output.write("[0] " + " ".join([ idxvocab[item] for item in s ]) + "\n")
        
        for temp in gen_temps:
            output.write("\nSentence generation (random; temperature = " + str(temp) + "):\n")
            for i in xrange(gen_num):
                s = mgen.generate_on_topic(sess, t, vocabxid[start_symbol], temp, cf.lm_sent_len+10, \
                    vocabxid[end_symbol])
                output.write("[" + str(i) + "] " +  " ".join([ idxvocab[item] for item in s ]) + "\n")

def gen_sent_on_doc(docs, tags, idxvocab, vocabxid, start_symbol, end_symbol, cf):
    topics, _ = tm.get_topics(sess, topn=topn)
    topics = [ " ".join([idxvocab[w] for w in t]) for t in topics ]
    doc_text = [ item.replace("\t", "\n") for item in codecs.open(args.input_doc, "r", "utf-8").readlines() ]
    output = codecs.open(args.gen_sent_on_doc, "w", "utf-8")
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mgen = LM(is_training=False, vocab_size=len(idxvocab), batch_size=1, num_steps=1, config=cf, \
            reuse_conv_variables=True)

    for d in range(len(docs)):
        output.write("\n" + "="*100 + "\n")
        output.write("Doc " +  str(d) +":\n")
        output.write(doc_text[d])

        doc, _, _, t, _ = get_batch_doc(docs, None, tags, d, cf.doc_len, cf.tag_len, 1, vocabxid[pad_symbol])
        best_topics, best_words = mgen.get_topics_on_doc(sess, doc, t, topn)
        
        output.write("\nRepresentative topics:\n")
        output.write("\n".join([ ("[%.3f] %s: %s" % (item[1],str(item[0]).zfill(3),topics[item[0]])) \
            for item in best_topics ]) + "\n")

        output.write("\nRepresentative words:\n")
        output.write("\n".join([ ("[%.3f] %s" % (item[1], idxvocab[item[0]])) for item in best_words ]) + "\n")

        output.write("\nSentence generation (greedy; argmax):" + "\n")
        s = mgen.generate_on_doc(sess, doc, t, vocabxid[start_symbol], 0, cf.lm_sent_len+10, vocabxid[end_symbol])
        output.write("[0] " + " ".join([ idxvocab[item] for item in s ]) + "\n")

        for temp in gen_temps:
            output.write("\nSentence generation (random; temperature = " + str(temp) + "):\n")

            for i in xrange(gen_num):
                s = mgen.generate_on_doc(sess, doc, t, vocabxid[start_symbol], temp, cf.lm_sent_len+10, \
                    vocabxid[end_symbol])
                output.write("[" + str(i) + "] " + " ".join([ idxvocab[item] for item in s ]) + "\n")
######
#main#
######

#load the vocabulary
vocab = cPickle.load(open(os.path.join(args.model_dir, "vocab.pickle")))
idxvocab, tm_ignore, dummy_symbols = vocab[0], vocab[1], vocab[2]
pad_symbol, start_symbol, end_symbol = dummy_symbols[0], dummy_symbols[1], dummy_symbols[2]
vocabxid = dict([(y,x) for x,y in enumerate(idxvocab)])

#load config
cf_dict = cPickle.load(open(os.path.join(args.model_dir, "config.pickle")))
if "num_classes" not in cf_dict:
    cf_dict["num_classes"] = 0
if "num_tags" not in cf_dict:
    cf_dict["num_tags"] = 0
    cf_dict["tag_len"] = 0
    cf_dict["tag_embedding_size"] = 0
ModelConfig = namedtuple("ModelConfig", " ".join(cf_dict.keys()))
cf = ModelConfig(**cf_dict)

#parse and collect the documents
if args.input_doc:
    sents, docs, docids, stats = gen_data(vocabxid, dummy_symbols, tm_ignore, args.input_doc, \
        cf.tm_sent_len, cf.lm_sent_len, cf.verbose, False)
    #print documents statistics
    print "Vocab size =", len(idxvocab)
    print_corpus_stats("Documents statistics", sents, docs, stats)

#collect the labels
#labels given for documents
if args.input_label:
    labels = [ int(item) for item in open(args.input_label).readlines() ]
else:
    labels = None

#collect the tags
if cf.num_tags > 0:
    if not args.input_tag:
        sys.stderr.write("Error: Saved model is trained with document tags; " + \
            "please specify tag metadata using --input_tag\n")
        raise SystemExit

    tagxid = cPickle.load(open(os.path.join(args.model_dir, "tag.pickle")))
    tags = [ [ tagxid[t] for t in line.strip().split("\t") if t in tagxid ] \
        for line in codecs.open(args.input_tag, "r", "utf-8").readlines() ]
else:
    tags = None

with tf.Graph().as_default(), tf.Session() as sess:
    initializer = tf.contrib.layers.xavier_initializer(seed=cf.seed)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        tm = TM(is_training=False, vocab_size=len(idxvocab), batch_size=cf.batch_size, \
            num_steps=cf.tm_sent_len, num_classes=cf.num_classes, config=cf) if cf.topic_number > 0 else None
        lm = LM(is_training=False, vocab_size=len(idxvocab), batch_size=cf.batch_size, \
            num_steps=cf.lm_sent_len, config=cf, reuse_conv_variables=True) if cf.rnn_hidden_size > 0 else None

    #load tensorflow model
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(args.model_dir, "model.ckpt"))

    #compute topic distribution of input documents
    if args.output_topic_dist:
        if args.input_doc == None:
            sys.stderr.write("Error: --output_topic_dist option requires --input_doc\n")
            raise SystemExit
        compute_dt_dist(docs[0], labels, tags, tm, cf.doc_len, cf.batch_size, vocabxid[pad_symbol], idxvocab, \
            args.output_topic_dist)

    #print topics
    if args.output_topic:
        topics, entropy = tm.get_topics(sess, topn=topn)
        output = codecs.open(args.output_topic, "w", "utf-8")
        for ti, t in enumerate(topics):
            output.write(" ".join([ idxvocab[item] for item in t ]) + "\n")

    #output tag embeddings
    if args.output_tag_embedding and cf.num_tags > 0:
        np.save(open(args.output_tag_embedding, "w"), sess.run(tm.tag_embedding))

    #compute test perplexities
    if args.print_perplexity:
        if args.input_doc == None:
            sys.stderr.write("Error: --print_perplexity requires --input_doc\n")
            raise SystemExit
        run_epoch(sents, docs, tags, (tm, lm), vocabxid[pad_symbol], cf, idxvocab)

    #compute classification accuracy
    if args.print_acc:
        if tm == None:
            sys.stderr.write("Error: Saved model does not have a topic model component; --print_acc ignored\n")
            raise SystemExit
        if cf.num_classes == 0:
            sys.stderr.write("Error: Saved topic model is not a supervised topic model; --print_acc ignored\n")
            raise SystemExit
        if labels == None or args.input_doc == None:
            sys.stderr.write("Error: --print_acc requires --input_doc and --input_label\n")
            raise SystemExit
        run_epoch_doc(docs[0], labels, tags, tm, vocabxid[pad_symbol], cf)

    #generate sentences conditioned on topics
    if args.gen_sent_on_topic:
        gen_sent_on_topic(idxvocab, vocabxid, start_symbol, end_symbol, cf)

    #generate sentences conditioned on documents
    if args.gen_sent_on_doc:
        if args.input_doc == None:
            sys.stderr.write("Error: --gen_sent_on_doc option requires --input_doc\n")
            raise SystemExit
        gen_sent_on_doc(docs[0], tags, idxvocab, vocabxid, start_symbol, end_symbol, cf)
