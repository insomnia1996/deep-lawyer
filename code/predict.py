# -*-coding:utf-8-*-
import json
import jieba
import thulac
from sklearn.externals import joblib
from config import *
from keras.models import load_model
from tools import *
import train
import numpy as np
import os
import heapq
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def text_to_npy(text):
	#process text to predicted
	#分词
	textlst=jieba.cut(text)
	text=" ".join(textlst)
	#词嵌入模型
	
	#语法树构建
	
	#

	#
	facts = np.load(os.path.join(data_path, "predicted/" + "pred_singlefact.npy"))
	pair_front_word = np.load(os.path.join(data_path,"predicted/" + "pred_pair_front_id.npy"))
	pair_last_word = np.load(os.path.join(data_path, "predicted/" +"pred_pair_last_id.npy"))
	pair_front_isword = np.load(os.path.join(data_path, "predicted/" + "_pair_front_word.npy"))
	pair_last_isword = np.load(os.path.join(data_path, "predicted/" + "_pair_last_word.npy"))
	pair_front_num = np.load(os.path.join(data_path, "predicted/" +"_pair_front_num.npy"))
	pair_last_num = np.load(os.path.join(data_path, "predicted/" + "_pair_last_num.npy"))
	pair_front_isnum = np.load(os.path.join(data_path, "predicted/" +"_pair_front_dig.npy"))
	pair_last_isnum = np.load(os.path.join(data_path, "predicted/" + "_pair_last_dig.npy"))
	return facts, pair_front_word, pair_last_word, pair_front_isword, pair_last_isword, pair_front_num, pair_last_num, pair_front_isnum, pair_last_isnum


def load_data(file_name):
	#[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input,
	# pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input]
	facts = np.load(os.path.join(data_path, "generated/" + file_name + "_cuted_singlefact.npy"))
	#laws = np.load(os.path.join(data_path, "generated/" + file_name + "_cuted_law.npy"))
	#accus = np.load(os.path.join(data_path, "generated/" + file_name + "_cuted_accu.npy"))
	#term = np.load(os.path.join(data_path, "generated/" + file_name + "_cuted_term.npy"))
	pair_front_word = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_front_id.npy"))
	pair_last_word = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_last_id.npy"))
	pair_front_isword = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_front_word.npy"))
	pair_last_isword = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_last_word.npy"))
	pair_front_num = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_front_num.npy"))
	pair_last_num = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_last_num.npy"))
	pair_front_isnum = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_front_dig.npy"))
	pair_last_isnum = np.load(os.path.join(data_path, "generated/" + file_name + "_pair_last_dig.npy"))
	return facts, pair_front_word, pair_last_word, pair_front_isword, pair_last_isword, pair_front_num, pair_last_num, pair_front_isnum, pair_last_isnum


class Predictor(object):
	def __init__(self):
		self.tfidf = joblib.load(os.path.join(data_path, 'generated/tfidf.model'))
		self.law = joblib.load(os.path.join(data_path, 'generated/law.model'))
		self.accu = joblib.load(os.path.join(data_path, 'generated/accu.model'))
		self.time = joblib.load(os.path.join(data_path, 'generated/time.model'))
		self.batch_size = 1

		self.cut = thulac.thulac(seg_only=True)

	def predict_law(self, vec):
		y = self.law.predict(vec)
		return [int(y[0]) + 1]

	def predict_accu(self, vec):
		y = self.accu.predict(vec)
		return [int(y[0]) + 1]

	def predict_time(self, vec):

		y = self.time.predict(vec)[0]

		# 返回每一个罪名区间的中位数
		if y == 0:
			return -2
		if y == 1:
			return -1
		if y == 2:
			return 120
		if y == 3:
			return 102
		if y == 4:
			return 72
		if y == 5:
			return 48
		if y == 6:
			return 30
		if y == 7:
			return 18
		else:
			return 6

	def predict(self, content):
		fact = self.cut.cut(content, text=True)

		vec = self.tfidf.transform([fact])
		ans = {}

		ans['accusation'] = self.predict_accu(vec)
		ans['articles'] = self.predict_law(vec)
		ans['imprisonment'] = self.predict_time(vec)

		print(ans)
		return ans


def svm_predict():
	predictor = Predictor()
	# alltext, accu_label, law_label, time_label = read_trainData('../data/small_data_train.json')

	with open(os.path.join(data_path, test_file_path), "r") as f:
		with open(os.path.join(data_path, "results/small_data_test_svm.json"), "w") as w:
			for line in f:
				# print(line)
				predict_res = predictor.predict(json.loads(line)['fact'])
				# print(predict)
				w.write(json.dumps(predict_res) + "\n")
	print("endpredicting ")


def han_predict():
	config=Config()
	#filepath = "cnnmodel.h5"
	filepath="finalmodel_test.h5"
	model = load_model(os.path.join(data_path, "generated/" + filepath), custom_objects={'AttLayer': train.AttLayer})
	#print(model.input_shape) #(none,512):singlefact,不拆分短句
	model.summary()
	test_facts, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum = load_data("new_data_test")
	test_accu_list = [list(range(0, config.num_accu_liu))] * test_facts.shape[0]
	test_law_list = [list(range(0, config.num_law_liu))] * test_facts.shape[0]
	test_term_list = [list(range(0, config.num_term_liu))] * test_facts.shape[0]
	test_accus= np.asarray(test_accu_list, dtype='int32')
	test_laws = np.asarray(test_law_list, dtype='int32')
	test_term = np.asarray(test_term_list, dtype='int32')
	accu_pred = model.predict([test_facts, test_accus, test_laws, test_term, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum])
	print(accu_pred[0].shape,accu_pred[1].shape,accu_pred[2].shape)#article,accusation,term
	for i in range(len(accu_pred)):
		accu_pred[i] = np.argmax(accu_pred[i], axis=1).tolist()
	accu_pred=list(map(list,zip(*accu_pred)))
	with open(os.path.join(data_path, "results/new_data_test_cuted.json"), "w") as w:
		predict_res={}
		for accu in (accu_pred):
			
			predict_res['accusation'] = accu[1]
			predict_res['articles'] = accu[0]
			predict_res['imprisonment'] = accu[2]
			w.write(json.dumps(predict_res) + "\n")


def cnn_predict():
	filepath="cnnmodel.h5"
	model = load_model(os.path.join(data_path, "generated/" + filepath), custom_objects={'AttLayer': train.AttLayer})
	#print(model.input_shape) #(none,512):singlefact,不拆分短句
	model.summary()
	test_facts, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum = load_data("new_data_test")
	accu_pred = model.predict(test_facts)
	print(accu_pred[0].shape,accu_pred[1].shape,accu_pred[2].shape)#accu,article,term (26766,119)(26766,103)(26766,11)
	for i in range(len(accu_pred)):
		accu_pred[i] = np.argmax(accu_pred[i], axis=1).tolist()
	accu_pred=list(map(list,zip(*accu_pred)))
	

	with open(os.path.join(data_path, "results/new_data_test_cnn.json"), "w") as w:
		predict_res={}
		for accu in (accu_pred):
			
			predict_res['accusation'] = accu[0]
			predict_res['articles'] = accu[1]
			predict_res['imprisonment'] = accu[2]
			w.write(json.dumps(predict_res) + "\n")


def cnn_predict2():
	filepath="cnnmodel.h5"
	model = load_model(os.path.join(data_path, "generated/" + filepath), custom_objects={'AttLayer': train.AttLayer})
	#print(model.input_shape) #(none,512):singlefact,不拆分短句
	#model.summary()
	test_facts, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum = load_data("new_data_test")
	accu_pred = model.predict(test_facts)
	new_pred=[]
	print(accu_pred[0].shape,accu_pred[1].shape,accu_pred[2].shape)#accu,article,term (26766,119)(26766,103)(26766,11)
	for i in range(len(accu_pred)):
		new_pred.append([])
		for j in range(len(accu_pred[i])):
			new_pred[i].append([])
			re1=heapq.nlargest(2, accu_pred[i][j])
			re2 = list(map(list(accu_pred[i][j]).index, re1))
			re1=[float(m) for m in re1]
			new_pred[i][j]=[(re1[0],re2[0]),(re1[1],re2[1])]#prob1,index1,prob2,index2
	new_pred=list(map(list,zip(*new_pred)))#矩阵转置    
	print(len(new_pred),len(new_pred[0]),len(new_pred[0][0]))
	
	with open(os.path.join(data_path, "results/new_data_test_cnn2.json"), "w") as w:
		predict_res={}
		for accu in (new_pred):
			
			predict_res['accusation'] = accu[0]
			predict_res['articles'] = accu[1]
			predict_res['imprisonment'] = accu[2]
			w.write(json.dumps(predict_res) + "\n")

def final_predict2():
	config=Config()
	filepath="finalmodel_test.h5"
	model = load_model(os.path.join(data_path, "generated/" + filepath), custom_objects={'AttLayer': train.AttLayer})
	#print(model.input_shape) #(none,512):singlefact,不拆分短句
	model.summary()
	test_facts, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum = load_data("new_data_test")
	test_accu_list = [list(range(0, config.num_accu_liu))] * test_facts.shape[0]
	test_law_list = [list(range(0, config.num_law_liu))] * test_facts.shape[0]
	test_term_list = [list(range(0, config.num_term_liu))] * test_facts.shape[0]
	test_accus= np.asarray(test_accu_list, dtype='int32')
	test_laws = np.asarray(test_law_list, dtype='int32')
	test_term = np.asarray(test_term_list, dtype='int32')
	accu_pred = model.predict([test_facts, test_accus, test_laws, test_term, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum])
	print(accu_pred[0].shape,accu_pred[1].shape,accu_pred[2].shape)#article,accusation,term (26766,119)(26766,103)(26766,11)
	new_pred=[]
	for i in range(len(accu_pred)):
		new_pred.append([])
		for j in range(len(accu_pred[i])):
			new_pred[i].append([])
			re1=heapq.nlargest(2, accu_pred[i][j])
			re2 = list(map(list(accu_pred[i][j]).index, re1))
			re1=[float(m) for m in re1]
			new_pred[i][j]=[(re1[0],re2[0]),(re1[1],re2[1])]#prob1,index1,prob2,index2
	new_pred=list(map(list,zip(*new_pred)))#矩阵转置    
	print(len(new_pred),len(new_pred[0]),len(new_pred[0][0]))
	
	with open(os.path.join(data_path, "results/new_data_test_cuted2.json"), "w") as w:
		predict_res={}
		for accu in (new_pred):
			
			predict_res['accusation'] = accu[1]
			predict_res['articles'] = accu[0]
			predict_res['imprisonment'] = accu[2]
			w.write(json.dumps(predict_res) + "\n")

def app_predict(text):
	config=Config()
	#filepath = "cnnmodel.h5"
	filepath="finalmodel_test.h5"
	model = load_model(os.path.join(data_path, "generated/" + filepath), custom_objects={'AttLayer': train.AttLayer})
	#print(model.input_shape) #(none,512):singlefact,不拆分短句
	model.summary()
	test_facts, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum = text_to_npy(text)
	test_accu_list = [list(range(0, config.num_accu_liu))] * test_facts.shape[0]
	test_law_list = [list(range(0, config.num_law_liu))] * test_facts.shape[0]
	test_term_list = [list(range(0, config.num_term_liu))] * test_facts.shape[0]
	test_accus= np.asarray(test_accu_list, dtype='int32')
	test_laws = np.asarray(test_law_list, dtype='int32')
	test_term = np.asarray(test_term_list, dtype='int32')
	accu_pred = model.predict([test_facts, test_accus, test_laws, test_term, test_pair_front_word, test_pair_last_word, test_pair_front_isword, test_pair_last_isword, test_pair_front_num, test_pair_last_num, test_pair_front_isnum, test_pair_last_isnum])
	print(accu_pred[0].shape,accu_pred[1].shape,accu_pred[2].shape)#article,accusation,term
	for i in range(len(accu_pred)):
		accu_pred[i] = np.argmax(accu_pred[i], axis=1).tolist()
	accu_pred=list(map(list,zip(*accu_pred)))
	with open(os.path.join(data_path, "results/predicted.json"), "w") as w:
		predict_res={}
		for accu in (accu_pred):
			
			predict_res['accusation'] = accu[1]
			predict_res['articles'] = accu[0]
			predict_res['imprisonment'] = accu[2]
			w.write(json.dumps(predict_res) + "\n")

if __name__ == "__main__":
	final_predict2()
	#cnn_predict2()
			