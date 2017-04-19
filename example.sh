#!/bin/bash

#train a word2vec model using gensim. This step is *optional*, you'll only need to do this if you want to initialise the model with pre-trained embeddings
#word2vec model settings are all in the python file itself (word2vec.py)
python word2vec_train.py

#train a model; configurations/hyper-parameters are all defined in tdlm_config.py
python tdlm_train.py

#all test inferences are invoked with tdlm_test.py
#compute language and topic model perplexity
python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --print_perplexity

#print topics (to topics.txt)
python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --output_topic topics.txt

#infer topic distribution in documents (saved as a npy file)
python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --output_topic_dist topic-dist.npy

#generate sentences conditioned on topics
python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --gen_sent_on_topic topic-sents.txt
