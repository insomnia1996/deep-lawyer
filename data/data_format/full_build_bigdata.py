# -*-coding:utf-8-*-
import os
import sys 
sys.path.append("../..") 
import json
from config import *
from itertools import chain
from collections import Counter
from pprint import pprint
import numpy as np

UNK = 0
config = Config()

stop_list = ["某", "×", "）", "（ ", "《", "》", "“", "”", "、"]


def process_sent(fact, separator, word2id):
    word_lens = []
    sents = []
    word_list = fact.split()
    
    start = 0
    end = 1
    
    for word in word_list:
        if word in separator:
            sents.append(word_list[start:end])
            start = end
            end = end + 1
        else:
            end += 1
    sents.append(word_list[start:end])
    new_sents = []
    for sent in sents:
        new_words = []
        for word in sent:
            ok = True
            for stop in stop_list:
                if stop in word:
                    ok = False
                    break
            if ok is True:
                new_words.append(word)
        new_sents.append(new_words)
    sents = new_sents

    sent_len = len(sents)
    for sent in sents:
        # if len(sent) > 100:
        #     print(sent)
        word_lens.append(len(sent))
        
    sents = [sent[:config.num_words] for i, sent in enumerate(sents) if i < config.num_sents]
        # print("sents : ", sents)
        # #每句短句长64，每短句词长64.
        #长度不够python会自动截断[:config.num_words]
    sent_matrix = [[0] * config.num_words for i in range(config.num_sents)]
        
        # print(sent_matrix)
    for i, sent in enumerate(sents):
        
        for j, word in enumerate(sent):
            try:
                sent_matrix[i][j] = word2id[word]
            except:
                sent_matrix[i][j] = UNK
    # pprint(sents)
    # pprint(sent_matrix)
    return sent_matrix, sent_len, word_lens


def process_single_sent(fact, separator, word2id):
    word_lens = []
    new_words = []
    word_list = fact.split()
    delist = stop_list + separator
    for word in word_list:
        ok = True
        for stop in delist:
            if stop in word:
                ok = False
                break
        if ok is True:
            new_words.append(word)

    lenth = len(new_words)

    new_words = new_words[:config.total_num]

    word_matrix = [0] * config.total_num

    # print(sent_matrix)
    for i, word in enumerate(new_words):
        try:
            word_matrix[i] = word2id[word]
        except:
            word_matrix[i] = UNK
    # pprint(sents)
    # pprint(sent_matrix)
    return word_matrix, lenth



def load_label_vocab():
    print("Now is loading law txt")
    id2law = {}
    law2id = {}
    id2accu = {}
    accu2id = {}
    with open("../useful/law.txt", "r",encoding="utf-8") as f:
        for i, line in enumerate(f):
            id2law[i] = line
            law2id[line] = i
    with open("../useful/accu.txt", "r",encoding="utf-8") as f:
        for i, line in enumerate(f):
            id2accu[i] = line
            accu2id[line] = i

    return id2law, law2id, id2accu, accu2id


def load_vocab():
    print("now is loading word2id dic")
    with open(os.path.join(data_path, "useful/new_word2id.json"), "r",encoding="utf-8") as f:
        word2id = json.load(f)
    
    return word2id



def gen_data(file_name):
    word2id = load_vocab()
    id2law, law2id, id2accu, accu2id = load_label_vocab()
    separator = ['，', '。', '：', '；']
    file_path = os.path.join(data_path,"useful/"+ file_name + ".json")
    facts = []
    laws = []
    accus = []
    terms = []
    times = []
    term_cates = []
    singlefacts = []
    tot_sent_lens = []
    tot_word_lens = []
    tot_single_lens = []
    totaldata = 0
    with open(file_path, "r",encoding="utf-8") as f:
        for i, line in enumerate(f):
            totaldata += 1
            if (totaldata % 10000 == 0):
                print (totaldata)
            sample = json.loads(line)
            fact = sample['fact_cut']
            law = sample['law']
            accu = sample['accu']
            term = sample['term']
            time = sample['time']
            term_cate = sample['term_cate']
            # print(accu)
            # print(law)
            
            # print("fact :", fact)
            sents, sent_len, word_lens = process_sent(fact, separator, word2id)
            singlesent, lenth = process_single_sent(fact, separator, word2id)
            # sents = np.array(sents)
            # if sents.sum() == 0:
            #     continue
            singlefacts.append(singlesent)
            facts.append(sents)
            laws.append(law)
            accus.append(accu)
            terms.append(term)
            times.append(time)
            term_cates.append(term_cate)
            # pprint(sents)
            
            tot_sent_lens.append(sent_len)
            tot_word_lens.append(word_lens)
            tot_single_lens.append(lenth)

    tot_word_lens = list(chain(*tot_word_lens))
    sent_len_counter = Counter(tot_sent_lens)
    word_len_counter = Counter(tot_word_lens)
    single_len_counter = Counter(tot_single_lens)
    #print(sents)
    print("sents : ")
    for s, freq in dict(sent_len_counter.most_common(len(sent_len_counter))).items():
        print(s, " : ", freq)
        print("words : ")
        for w, freq in dict(word_len_counter.most_common(len(word_len_counter))).items():
            print(w, " : ", freq)
    print("single : ")
    for ss, freq in dict(single_len_counter.most_common(len(single_len_counter))).items():
        print(ss, " : ", freq)
    
    
    print("Now is saving processed", file_name)
    '''
    np_facts=np.array(facts)
    np_laws=np.array(laws)
    np_accus=np.array(accus)
    print(np_facts.shape,np_laws.shape,np_accus.shape)
    '''
    np.save(os.path.join(data_path, "generated/" + file_name + "_fact.npy"), facts)
    np.save(os.path.join(data_path, "generated/" + file_name + "_singlefact.npy"), singlefacts)
    np.save(os.path.join(data_path, "generated/" + file_name + "_law.npy"), laws)
    np.save(os.path.join(data_path, "generated/" + file_name + "_accu.npy"), accus)
    np.save(os.path.join(data_path, "generated/" + file_name + "_term.npy"), terms)
    np.save(os.path.join(data_path, "generated/" + file_name + "_time.npy"), times)
    np.save(os.path.join(data_path, "generated/" + file_name + "_term_cate.npy"), term_cates)
    
if __name__ == "__main__":
    gen_data("new_data_train_cuted")
    gen_data("new_data_test_cuted")

