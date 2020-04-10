#交互界面将案情描述预处理
# -*-coding:utf-8-*-
import os
data_path = "D:/deep-lawyer/data/"
#test_file_path = "processed/small_data_test_clear.json"
#train_file_path = os.path.join(data_path, "processed/small_data_train_clear.json")
test_file_path = "basic/small_data_test.json"
train_file_path = os.path.join(data_path, "basic/small_data_train.json")
#law_file_name = "law_clear.txt"
#accu_file_name = "accu_clear.txt"
law_file_name = "new_law.txt"
accu_file_name = "new_accu.txt"
law_path = os.path.join(data_path, law_file_name)
accu_path = os.path.join(data_path, accu_file_name)
word_vect_path = os.path.join(data_path, "basic/sgns.renmin.word")


class Config:
    num_sents = 64
    num_words = 64
    total_num = 512
    total_pair = 128
    batch_size = 128
    hid_size = 256
    word_embed_size = 200
#    word_vocab_size = 408388
    word_vocab_size = 26883
#   word_vocab_size = 26735
    embedding_path = os.path.join(data_path, "word_matrix.npy")
    law2accu = os.path.join(data_path, "law2accu.npy")
    accu2law = os.path.join(data_path, "accu2law.npy")
    law2term = os.path.join(data_path, "law2term.npy")
    accu2term = os.path.join(data_path, "accu2term.npy")
    num_law = 183
    num_law_clear = 183
#    num_law_clear = 181
    num_accu = 202
    num_accu_clear = 202
#    num_accu_clear = 193
#    num_accu_liu = 119
#    num_law_liu = 103
#    num_term_liu = 11
    num_accu_liu = 130
    num_law_liu = 118
    num_term_liu = 11
    epochs = 16
    attention_dim = 256
