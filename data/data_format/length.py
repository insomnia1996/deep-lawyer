import numpy as np
import json,re
from matplotlib import pyplot as plt
train_text=[]
f=open('../useful/new_data_train_cuted.json','r',encoding='utf-8')
while True:
	line=f.readline()
	if line:
		dic=json.loads(line,strict=False)
		pairlst=dic['fact_cut']
		train_text.append(pairlst)
	else:
		break



sentence_length = [len(x)/2 for x in train_text] #train_text是train.csv中每一行分词之后的数据

plt.hist(sentence_length,bins=5000,normed=1,cumulative=True)
plt.xlim(0,512)
plt.ylim(0,1)
plt.show()
