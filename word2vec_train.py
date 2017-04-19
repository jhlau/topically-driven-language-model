import gensim.models as g
import logging
import os

#parameters
output_dir="word2vec"
input_doc="data/toy-train.txt"

#main
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docs = g.word2vec.LineSentence(input_doc)
m = g.Word2Vec(docs, size=50, alpha=0.025, window=5, min_count=2, \
    sample=1e-5, workers=4, min_alpha=0.0001, sg=1, hs=0, negative=5, iter=100)

#save model
m.save(output_dir + "/skipgram.bin")
