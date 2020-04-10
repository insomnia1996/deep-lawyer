import json

file1 = open("../useful/new_data_train_cuted.json", "r",encoding="utf-8")
file2 = open("../useful/new_data_test_cuted.json", "r",encoding="utf-8")

total = 0
wordlist = {}

for line in file1.readlines():
    sample = json.loads(line, strict=False)
    total += 1
    fact = sample['fact_cut']
    fact = fact.split()
    for word in fact:
        if word in wordlist:
            wordlist[word] += 1
        else:
            wordlist[word] = 1
print (total)

total = 0
for line in file2.readlines():
    sample = json.loads(line, strict=False)
    total += 1
    fact = sample['fact_cut']
    fact = fact.split()
    for word in fact:
        if word in wordlist:
            wordlist[word] += 1
        else:
            wordlist[word] = 1
print (total)

total = 1
finallist = {}
for word in wordlist:
    if (wordlist[word] >= 25):
        finallist[word] = total
        total += 1

print (total)
json.dump(finallist, open('new_word2id.json', 'w'))

#print finallist

