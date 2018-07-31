# Requirements
- python2.7 (development python3 code available in python3 branch; code still requires testing)
- gensim: pip install gensim
- tensorflow 0.8-0.12

# Data Format
- One line per document
- Sentences are delimited by tabs in each document
- See examples in data/
- [ACL2017 Paper dataset (AP News, BNC and IMDB)](https://ibm.box.com/s/ls61p8ovc1y87w45oa02zink2zl7l6z4)

# Running the code (example.sh)

#### Train a word2vec model using gensim. This step is *optional*, you'll only need to do this if you want to initialise TDLM with pre-trained embeddings. word2vec model settings are in the python file (word2vec.py)

`python word2vec_train.py`

#### Train a model; configurations/hyper-parameters are defined in tdlm_config.py

`python tdlm_train.py`

#### All test inferences are invoked with tdlm_test.py. E.g. to compute language and topic model perplexity

`python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --print_perplexity`

#### Print topics (to topics.txt)

`python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --output_topic topics.txt`

#### Infer topic distribution in documents (saved as a npy file)

`python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --output_topic_dist topic-dist.npy`

#### Generate sentences conditioned on topics

`python tdlm_test.py -m output/toy-model/ -d data/toy-valid.txt --gen_sent_on_topic topic-sents.txt`

#### tdlm_test.py arguments:

```
usage: tdlm_test.py [-h] -m MODEL_DIR [-d INPUT_DOC] [-l INPUT_LABEL]
                    [-t INPUT_TAG] [--print_perplexity] [--print_acc]
                    [--output_topic OUTPUT_TOPIC]
                    [--output_topic_dist OUTPUT_TOPIC_DIST]
                    [--output_tag_embedding OUTPUT_TAG_EMBEDDING]
                    [--gen_sent_on_topic GEN_SENT_ON_TOPIC]
                    [--gen_sent_on_doc GEN_SENT_ON_DOC]

Given a trained TDLM model, perform various test inferences

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model_dir MODEL_DIR
                        directory of the saved model
  -d INPUT_DOC, --input_doc INPUT_DOC
                        input file containing the test documents
  -l INPUT_LABEL, --input_label INPUT_LABEL
                        input file containing the test labels
  -t INPUT_TAG, --input_tag INPUT_TAG
                        input file containing the test tags
  --print_perplexity    print topic and language model perplexity of the input
                        test documents
  --print_acc           print supervised classification accuracy
  --output_topic OUTPUT_TOPIC
                        output file to save the topics (prints top-N words of
                        each topic)
  --output_topic_dist OUTPUT_TOPIC_DIST
                        output file to save the topic distribution of input
                        docs (npy format)
  --output_tag_embedding OUTPUT_TAG_EMBEDDING
                        output tag embeddings to file (npy format)
  --gen_sent_on_topic GEN_SENT_ON_TOPIC
                        generate sentences conditioned on topics
  --gen_sent_on_doc GEN_SENT_ON_DOC
                        generate sentences conditioned on input test documents
```

# Publication

Jey Han Lau, Timothy Baldwin and Trevor Cohn (2017). [Topically Driven Neural Language Model](http://aclweb.org/anthology/P/P17/P17-1033.pdf). In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), Vancouver, Canada, pp. 355--365.
