#word2id_num:23877
#test_line:26766
#train_line:115472
import json,csv
import re,os,sys
import matplotlib.pyplot as plt
import codecs
BAR1=25#word_mincount
BAR2=500#seq_maxlength
def word2id(file):
	infile=open(file,'r',encoding='utf-8')
	worddic={}
	cnt=0
	sentence_len=[]
	while True:
		line=infile.readline()
		if line:
			datadic=json.loads(line)
			fact_cut=datadic['fact_cut']
			fact_lst=re.split(' ',fact_cut)
			seq_len=len(fact_lst)
			sentence_len.append(seq_len)                     
			for item in fact_lst:
				if item not in worddic:
					worddic[item]=[cnt,1] #word id , word count
					cnt+=1
				else:
					worddic[item][1]+=1
					#print("word already exists in the dictionary.\n")
		else:
			break
	cnt=0
	worddict={}
	for item in worddic:
		if worddic[item][1]>=BAR1:
			cnt+=1
			worddict[item]=cnt
	print("total id count: %d\n" %cnt)
	json_str=json.dumps(worddict)
	with open('../useful/word2id.json','w',encoding='utf-8') as outfile:
		outfile.write(json_str)
	#matplotlib shows sentence length
	'''
	plt.hist(sentence_len,1000,normed=1,cumulative=True)
	plt.xlim(0,500)
	plt.show()
	'''
def sent_w2i(sentence):
	f=open('../useful/word2id.json','r',encoding='utf-8')
	worddic=json.load(f)
	#print(type(worddic))
	sentlst=re.split(' ',sentence)
	sentidlst=[]
	for item in sentlst:
		if item in worddic:
			sentidlst.append(worddic[item])
		#else:
			#pass
			#print("Word not in training set word dictionary!")
	return sentidlst

# 转换文件格式
def json2csv_file(path):
	jsonData = codecs.open(path+'.json', 'r', 'utf-8')
	# csvfile = open(path+'.csv', 'w') # 此处这样写会导致写出来的文件会有空行
	# csvfile = open(path+'.csv', 'wb') # python2下
	csvfile = open(path+'.csv', 'w', newline='') # python3下
	writer = csv.writer(csvfile, delimiter='\t')
	flag=True
	for line in jsonData:
		dic = json.loads(line[0:-1])
		if flag:
			# 获取属性列表
			keys = list(dic.keys())
			print (keys)
			writer.writerow(keys) # 将属性列表写入csv中
			flag=False
			#if 1:
			tmp=sent_w2i(dic['fact_cut'])
			if len(tmp)<=BAR2:
				dic['fact_cut']=tmp
				# 读取json数据的每一行，将values数据一次一行的写入csv中
				writer.writerow(list(dic.values()))
		else:
			#if 1:
			tmp=sent_w2i(dic['fact_cut'])
			#if len(tmp)<=BAR2:
			print(len(tmp))
			dic['fact_cut']=tmp
			# 读取json数据的每一行，将values数据一次一行的写入csv中
			writer.writerow(list(dic.values()))
	jsonData.close()
	csvfile.close()


if __name__=='__main__':	
	#word2id("../useful/new_data_train_cuted.json")
	json2csv_file("../useful/new_data_train_cuted")
	#json2csv_file("123")
