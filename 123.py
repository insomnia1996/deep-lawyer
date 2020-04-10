import re,os,json
new_num=185228
#new_num=476955
#new_num=1111939
newdic={}
reldic={}
glbid=0
h=open('rel.txt','r')
while True:
	line=h.readline()
	if line:

		lst=re.split(',',line[:-1])
		reldic[lst[1].split('rel:')[1]]=int(lst[0].split('id:')[1])
		glbid+=1
	else:
		break
h.close()

f=open('data/basic/tst.json','r')
while True:
	line=f.readline()
	if line:
		try:
			dic=json.loads(line)
		except:
			print(line)

		for item in dic['pair']:
			if item['rel'] not in reldic:
				reldic[item['rel']]=glbid
				item['rel']=glbid
				glbid+=1
			else:
				item['rel']=reldic[item['rel']]
		newdic[dic['num']]=json.dumps(dic)
	else:
		break
f.close()
print(reldic)
#g=open('data/big_data_train_pair.json','w')
g=open('data/new_bigdata_test_pair.json','w')
for i in range(new_num):
	if i in newdic:
		g.write(newdic[i]+'\n')
	else:
		tmpdic={}
		tmpdic['num']=i
		tmpdic['pair']=[]
		g.write(json.dumps(tmpdic)+'\n')
g.close()
h=open('rel.txt','w')
nm=0
reldic2={}
for key,val in reldic.items():
    reldic2[val]=key
for i in range(len(reldic2)):
	h.write('id:%d,rel:%s\n' %(nm,reldic2[i]))
	nm+=1
h.close()
