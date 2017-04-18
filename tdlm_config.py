#preprocessing options
vocab_minfreq=2
vocab_maxfreq=0.001
stopwords="data/toy-stopword.txt"
tm_sent_len=3 #maximum sentence length for topic model; sentence passed the threshold is broken into multiple sequences
lm_sent_len=30 #maximum sentence length for language model
doc_len=300 #maximum document length; words passed the threshold is truncated/ignored

#training options
seed=1
batch_size=64
rnn_layer_size=1
rnn_hidden_size=60 #0 disables lstm
word_embedding_model=None
word_embedding_update=True #update word embedding for topic model
word_embedding_size=30 #setting ignored if word embeding model provided
topic_number=10 #0 disables topic model
topic_embedding_size=5 #topic output embedding size
tag_embedding_size=5
filter_sizes=[2]
filter_number=20 #topic input embedding size = filter_number * len(filter_sizes)
conv_activation="identity" #relu or identity
alpha=0.0
num_samples=0
epoch_size=1
learning_rate=0.001
tm_keep_prob=0.4
lm_keep_prob=0.6
max_grad_norm=5

#misc
save_model=True
verbose=True

#input/output
output_dir="output"
train_corpus="data/toy-train.txt"
train_target="data/toy-train-label.txt" #None if unsupervised
train_tag="data/toy-train-tag.txt" #None if no document tags
valid_corpus="data/toy-valid.txt"
valid_target="data/toy-valid-label.txt" #None if unsupervised
valid_tag="data/toy-valid-tag.txt" #None if no document tags
output_prefix="vmin%d_sup%d_tmslen%d_lmslen%d_dlen%d_seed%d_batch%d_lsize%d_hsize%d_wmodel%s_update%d_esize%d_topic%d_tesize%d_tagesize%d_fsizes%s_fnum%d_conv%s_alpha%.4f_sample%d_epoch%d_lr%.4f_tkp%.1f_lkp%.1f_maxgrad%d" % \
    (vocab_minfreq, ("train_target" in locals()), tm_sent_len, lm_sent_len, doc_len, seed, batch_size, rnn_layer_size, \
    rnn_hidden_size, (word_embedding_model.split("/")[-1].split(".")[0] if word_embedding_model != None else "none"), \
    word_embedding_update, word_embedding_size, topic_number, topic_embedding_size, tag_embedding_size, \
    "-".join([str(x) for x in filter_sizes]), filter_number, \
    conv_activation, alpha, num_samples, epoch_size, learning_rate, tm_keep_prob, lm_keep_prob, max_grad_norm)
