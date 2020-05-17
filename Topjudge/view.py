import json
from collections import Counter
import numpy as np

def getid(x, word2id):
    if x in word2id:
        wordid = word2id[x]
    else:
        wordid = 0
    return wordid

def getnum(x):
    x = int(x)
    num = [0] * 8
    pos = 0
    while x > 0:
        num[pos] = x % 10
        x /= 10
        pos += 1
        if (pos > 7):
            break
    if (x > 9):
        num = [9] * 8
    return num



if __name__ == '__main__':

    file1 = open("./processed/new_data_train_pair_new.json", "r")
    file2 = open("./processed/new_data_test_pair_new.json", "r")
    
    file3 = open("./new_word2id.json", 'r')
    word2id = json.load(file3)


    useful = [2,3,7,8,11,12,13,16,18,22,23,24,25]

    train_front = []
    train_front_dig = []
    train_front_word = []
    train_front_num = []
    
    train_last = []
    train_last_dig = []
    train_last_word = []
    train_last_num = []


    for i, line in enumerate(file1):
        sample = json.loads(line)
        total = 0
        frontlist = []
        front_dig = []
        front_word = []
        front_num = []
        
        lastlist = []
        last_dig = []
        last_word = []
        last_num = []
        
        for j in sample['pair']:
            rel = j['rel']
            if (rel in useful):
                x = j['former']
                y = j['latter']
                if (total >= 128):
                    break
                
                wordid = getid(x, word2id)
                frontlist.append(wordid)
                num = [0] * 8
                
                if x.isdecimal():
                    front_dig.append(1)
                    front_word.append(0)
#                    print 'dig'
                    num = getnum(x)
                else:
                    front_dig.append(0)
                    front_word.append(1)
                
                front_num.append(num)



                wordid = getid(y, word2id)
                lastlist.append(wordid)
                num = [0] * 8
                
                if y.isdecimal():
                    last_dig.append(1)
                    last_word.append(0)
#                    print (y)
                    num = getnum(y)
                else:
                    last_dig.append(0)
                    last_word.append(1)
                
                last_num.append(num)

                total += 1
#                print (total)
        while (total < 128):
            frontlist.append(0)
            front_dig.append(0)
            front_word.append(1)
            front_num.append([0] * 8)
            
            lastlist.append(0)
            last_dig.append(0)
            last_word.append(1)
            last_num.append([0] * 8)
            total += 1
        train_front.append(frontlist)
        train_front_dig.append(front_dig)
        train_front_word.append(front_word)
        train_front_num.append(front_num)
        
        train_last.append(lastlist)
        train_last_dig.append(last_dig)
        train_last_word.append(last_word)
        train_last_num.append(last_num)

    train_front_id = np.asarray(train_front)
    print (train_front_id.shape)
    train_front_dig = np.asarray(train_front_dig)
    print (train_front_dig.shape)
    train_front_word = np.asarray(train_front_word)
    print (train_front_word.shape)
    train_front_num = np.asarray(train_front_num)
    print (train_front_num.shape)

    train_last_id = np.asarray(train_last)
    print (train_last_id.shape)
    train_last_dig = np.asarray(train_last_dig)
    print (train_last_dig.shape)
    train_last_word = np.asarray(train_last_word)
    print (train_last_word.shape)
    train_last_num = np.asarray(train_last_num)
    print (train_last_num.shape)

    np.save('./generated/new_data_train_pair_front_id', train_front_id)
    np.save('./generated/new_data_train_pair_front_dig', train_front_dig)
    np.save('./generated/new_data_train_pair_front_word', train_front_word)
    np.save('./generated/new_data_train_pair_front_num', train_front_num)
    
    np.save('./generated/new_data_train_pair_last_id', train_last_id)
    np.save('./generated/new_data_train_pair_last_dig', train_last_dig)
    np.save('./generated/new_data_train_pair_last_word', train_last_word)
    np.save('./generated/new_data_train_pair_last_num', train_last_num)


    test_front = []
    test_front_dig = []
    test_front_word = []
    test_front_num = []

    test_last = []
    test_last_dig = []
    test_last_word = []
    test_last_num = []

    for i, line in enumerate(file2):
        sample = json.loads(line)
        total = 0
        frontlist = []
        front_dig = []
        front_word = []
        front_num = []
        
        lastlist = []
        last_dig = []
        last_word = []
        last_num = []
        
        for j in sample['pair']:
            rel = j['rel']
            if (rel in useful):
                x = j['former']
                y = j['latter']
                if (total >= 128):
                    break
                
                wordid = getid(x, word2id)
                frontlist.append(wordid)
                num = [0] * 8
                
                if x.isdecimal():
                    front_dig.append(1)
                    front_word.append(0)
                    num = getnum(x)
                else:
                    front_dig.append(0)
                    front_word.append(1)
            
                front_num.append(num)
                
                
                
                wordid = getid(y, word2id)
                lastlist.append(wordid)
                num = [0] * 8
                
                if y.isdecimal():
                    last_dig.append(1)
                    last_word.append(0)
                    num = getnum(y)
                else:
                    last_dig.append(0)
                    last_word.append(1)
                
                last_num.append(num)
                
                total += 1
                    
        while (total < 128):
            frontlist.append(0)
            front_dig.append(0)
            front_word.append(1)
            front_num.append([0] * 8)
            
            lastlist.append(0)
            last_dig.append(0)
            last_word.append(1)
            last_num.append([0] * 8)
            total += 1
        
        test_front.append(frontlist)
        test_front_dig.append(front_dig)
        test_front_word.append(front_word)
        test_front_num.append(front_num)
        
        test_last.append(lastlist)
        test_last_dig.append(last_dig)
        test_last_word.append(last_word)
        test_last_num.append(last_num)

    test_front_id = np.asarray(test_front)
    print (test_front_id.shape)
    test_front_dig = np.asarray(test_front_dig)
    print (test_front_dig.shape)
    test_front_word = np.asarray(test_front_word)
    print (test_front_word.shape)
    test_front_num = np.asarray(test_front_num)
    print (test_front_num.shape)

    test_last_id = np.asarray(test_last)
    print (test_last_id.shape)
    test_last_dig = np.asarray(test_last_dig)
    print (test_last_dig.shape)
    test_last_word = np.asarray(test_last_word)
    print (test_last_word.shape)
    test_last_num = np.asarray(test_last_num)
    print (test_last_num.shape)

    np.save('./generated/new_data_test_pair_front_id', test_front_id)
    np.save('./generated/new_data_test_pair_front_dig', test_front_dig)
    np.save('./generated/new_data_test_pair_front_word', test_front_word)
    np.save('./generated/new_data_test_pair_front_num', test_front_num)

    np.save('./generated/new_data_test_pair_last_id', test_last_id)
    np.save('./generated/new_data_test_pair_last_dig', test_last_dig)
    np.save('./generated/new_data_test_pair_last_word', test_last_word)
    np.save('./generated/new_data_test_pair_last_num', test_last_num)

