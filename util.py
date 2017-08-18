import codecs
import sys
import operator
import math
import re
import numpy as np
from collections import defaultdict

def init_embedding(model, idxvocab):
    word_emb = []
    for vi, v in enumerate(idxvocab):
        if v in model:
            word_emb.append(model[v])
        else:
            word_emb.append(np.random.uniform(-0.5/model.vector_size, 0.5/model.vector_size, [model.vector_size,]))
    return np.array(word_emb)

def pad(lst, max_len, pad_symbol):
    return lst + [pad_symbol] * (max_len - len(lst))

def get_batch(sents, docs, tags, idx, doc_len, sent_len, tag_len, batch_size, pad_id, remove_curr):
    x, y, m, d, t = [], [], [], [], []

    for docid, seqid, seq in sents[(idx*batch_size):((idx+1)*batch_size)]:
        if remove_curr:
            dw = docs[docid][:seqid] + docs[docid][(seqid+1):]
        else:
            dw = docs[docid]
        dw = [item for sublst in dw for item in sublst][:doc_len]
        d.append(pad(dw, doc_len, pad_id))
        x.append(pad(seq[:-1], sent_len, pad_id))
        y.append(pad(seq[1:], sent_len, pad_id))
        m.append([1.0]*(len(seq)-1) + [0.0]*(sent_len-len(seq)+1))
        if tags != None:
            t.append(pad(tags[docid][:tag_len], tag_len, pad_id))
        else:
            t.append([])

    for _ in range(batch_size - len(d)):
        d.append([pad_id]*doc_len)
        x.append([pad_id]*sent_len)
        y.append([pad_id]*sent_len)
        m.append([0.0]*sent_len)
        t.append([pad_id]*tag_len)

    return x, y, m, d, t


def get_batch_doc(docs, labels, tags, idx, max_len, tag_len, batch_size, pad_id):
    x = []
    for d in docs[(idx*batch_size):((idx+1)*batch_size)]:
        dx = pad([item for sublst in d for item in sublst][:max_len], max_len, pad_id)
        x.append(dx)
    s = len(x)
    for _ in range(batch_size-len(x)):
        x.append([pad_id]*max_len)

    #mask
    m = [1.0]*s + [0.0]*(batch_size-s)

    #labels
    y = None
    if labels != None:
        y = labels[(idx*batch_size):((idx+1)*batch_size)]
        for _ in range(batch_size-s):
            y.append(0)

    #tags
    t = []
    if tags != None:
        for e in tags[(idx*batch_size):((idx+1)*batch_size)]:
            t.append(pad(e, tag_len, pad_id))
    else:
        t = [ []*batch_size ]
    for _ in range(batch_size-len(t)):
        t.append([pad_id]*tag_len)
        

    return x, y, m, t, s

def update_vocab(symbol, idxvocab, vocabxid):
    idxvocab.append(symbol)
    vocabxid[symbol] = len(idxvocab) - 1 

def gen_vocab(dummy_symbols, corpus, stopwords, vocab_minfreq, vocab_maxfreq, verbose):
    idxvocab = []
    vocabxid = defaultdict(int)
    vocab_freq = defaultdict(int)
    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        for word in line.strip().split():
            vocab_freq[word] += 1
        if line_id % 1000 == 0 and verbose:
            sys.stdout.write(str(line_id) + " processed\r")
            sys.stdout.flush()

    #add in dummy symbols into vocab
    for s in dummy_symbols:
        update_vocab(s, idxvocab, vocabxid)

    #remove low fequency words
    for w, f in sorted(vocab_freq.items(), key=operator.itemgetter(1), reverse=True):
        if f < vocab_minfreq:
            break
        else:
            update_vocab(w, idxvocab, vocabxid)

    #ignore stopwords, frequent words and symbols for the document input for topic model
    stopwords = set([item.strip().lower() for item in open(stopwords)])
    freqwords = set([item[0] for item in sorted(vocab_freq.items(), key=operator.itemgetter(1), \
        reverse=True)[:int(float(len(vocab_freq))*vocab_maxfreq)]]) #ignore top N% most frequent words for topic model
    alpha_check = re.compile("[a-zA-Z]")
    symbols = set([ w for w in vocabxid.keys() if ((alpha_check.search(w) == None) or w.startswith("'")) ])
    ignore = stopwords | freqwords | symbols | set(dummy_symbols) | set(["n't"])
    ignore = set([vocabxid[w] for w in ignore if w in vocabxid])

    return idxvocab, vocabxid, ignore

def gen_data(vocabxid, dummy_symbols, ignore, corpus, tm_sent_len, lm_sent_len, verbose, remove_short):
    sents = ([], []) #tuple of (tm_sents, lm_sents); each element is [(doc_id, seq_id, seq)]
    docs = ([], []) #tuple of (tm_docs, lm_docs); each element is [ [[doc1seq1], [doc2seq2]], [doc2_seqs] ]
    sent_lens = [] #original sentence lengths
    doc_lens = [] #original document lengths
    docids = [] #original document IDs
    start_symbol = dummy_symbols[1]
    end_symbol = dummy_symbols[2]
    unk_symbol = dummy_symbols[3]

    for line_id, line in enumerate(codecs.open(corpus, "r", "utf-8")):
        tm_sents = [vocabxid[start_symbol]] #sentences for tm
        lm_sents = [] #sentences for lm
        for s in line.strip().split("\t"):
            sent = [vocabxid[start_symbol]]
            for w in s.strip().split():
                if w in vocabxid:
                    sent.append(vocabxid[w])
                    if (vocabxid[w] not in ignore):
                        tm_sents.append(vocabxid[w])
                else:
                    sent.append(vocabxid[unk_symbol])
            sent.append(vocabxid[end_symbol])
            lm_sents.append(sent)

        #ignore documents with no words
        if not remove_short or (len(tm_sents) > 1):
            docids.append(line_id)
            sent_lens.extend([len(item)-1 for item in lm_sents])
            doc_lens.append(len(tm_sents))

            #chop tm_sents into sequences of length tm_sent_len
            seq_id = 0
            doc_seqs = []
            for si in range(int(math.ceil(len(tm_sents) * 1.0 / tm_sent_len))):
                seq = tm_sents[si*tm_sent_len:((si+1)*tm_sent_len+1)]
                if len(seq) > 1:
                    sents[0].append((len(docs[0]), seq_id, seq))
                    doc_seqs.append(seq[1:])
                    seq_id += 1
            docs[0].append(doc_seqs)

            #chop each sentence exceeding lm_sent_len into multiple sequences for the language model
            seq_id = 0
            doc_seqs = []
            for s in lm_sents:
                for si in range(int(math.ceil(len(s) * 1.0 / lm_sent_len))):
                    seq = s[si*lm_sent_len:((si+1)*lm_sent_len+1)]
                    if len(seq) > 1:
                        sents[1].append((len(docs[1]), seq_id, seq))
                        doc_seqs.append([w for w in seq[1:] if w not in ignore]) #output sentence = seq[1:]
                        seq_id += 1
            docs[1].append(doc_seqs)

        if line_id % 1000 == 0 and verbose:
            sys.stdout.write(str(line_id) + " processed\r")
            sys.stdout.flush()

    return sents, docs, docids, (np.mean(sent_lens), max(sent_lens), min(sent_lens), \
        np.mean(doc_lens), max(doc_lens), min(doc_lens))

def gen_tag(corpus, pad_symbol):
    tagxid = {pad_symbol:0}
    for line in codecs.open(corpus, "r", "utf-8"):
        for t in line.strip().split("\t"):
            if t not in tagxid:
                tagxid[t] = len(tagxid)
    return tagxid

def print_corpus_stats(name, sents, docs, stats):
    print(name + ":")
    print("\tno. of docs =", len(docs[0]))
    if len(sents[0]) > 0:
        print("\ttopic model no. of sequences =", len(sents[0]))
        print("\ttopic model no. of tokens =", sum([ len(item[2])-1 for item in sents[0] ]))
        print("\toriginal doc mean len =", stats[3])
        print("\toriginal doc max len =", stats[4])
        print("\toriginal doc min len =", stats[5])
    if len(sents[1]) > 0:
        print("\tlanguage model no. of sequences =", len(sents[1]))
        print("\tlanguage model no. of tokens =", sum([ len(item[2])-1 for item in sents[1] ]))
        print("\toriginal sent mean len =", stats[0])
        print("\toriginal sent max len =", stats[1])
        print("\toriginal sent min len =", stats[2])

def compute_tp_fp(x, y):
    n = max( [max(x), max(y)] ) + 1
    tp, fp = np.zeros((n), dtype=np.float32), np.zeros((n), dtype=np.float32)
    for ai, a in enumerate(x):
        if a == y[ai]:
            tp[a] += 1.0
        else:
            fp[a] += 1.0

    return tp, fp

def micro_f1(sys, gold):
    stp, sfp = compute_tp_fp(sys, gold)
    gtp, gfp = compute_tp_fp(gold, sys)

    p = sum(stp) / (sum(stp) + sum(sfp))
    r = sum(gtp) / (sum(gtp) + sum(gfp))
    if p != 0.0 or r != 0.0:
        f = 2*p*r / (p+r)
    else:
        f = 0.0

    return f
