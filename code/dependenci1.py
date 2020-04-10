#!/usr/bin/python
# -*- coding: UTF-8 -*-
import json,re,os
#import jieba
from nltk.tree import Tree
import threading
from nltk.parse.stanford import StanfordParser,StanfordDependencyParser
nlist=['NN','NR','NT','CD','M']
vlist=['VV','VP','MD','COP']
#tstr="砚山县 人民 检察院 指控 ， 2016年 1月 14日 8日 时 许 ， 被告人 田某 窜 至 砚山县 阿 舍乡 阿 舍 村委会 白世泥村 周2 某 家 门前 将 一头 小 母 水牛 盗 走 。 后 田某 打算 将 牛某 至 开远市 羊街镇 卖 掉 。 2016年 1月 22日 中午 ， 田某 拉 牛 至 左 美果村 时 被 村民 拦住 ， 后 被 开远市 碑格 派出所 抓获 。 为 证明 指控 事实 成立 ， 公诉 机关 向 法庭 提交 了 物证 、 书证 、 证人 证言 、 被害人 陈述 、 被告人 的 供述 与 辩解 、 鉴定 意见 、 勘验 、 检查 笔录 等 证据 。 砚山县 人民 检察院 认为 ， 被告人 田某 目无国法 ， 盗窃 他人 财物 ， 数额 较 大 ， 其 行为 应 以 ×× 追究 刑事 责任 。 被告人 田某 认罪 态度 好 可 酌情 从轻 处罚 ， 建议 对 被告人 田某 在 ×× 至 一 年 八 个 月 幅度 内 量刑 ， 并 处 罚金 。"
tstr=" 李某 于 2012年 12月 1日 19时 许 ， 到 广州市 南沙区 新 垦 十四 涌 西面 机耕路 附近 的 薯丛 中 ， 用 随身 携带 的 摩托车 钥匙 将 被害人 赵某 的 两 轮 摩托车 （ 发动机 号 ： GP 100104121 ， 车架号 ： LCS5BPK H 458762310 ， 价 值 人民币 3318 元 ） 的 车头 锁 打开 后 盗走 。 "

#path_to_jar = '../stanford-parser-full-2018-02-27/stanford-parser.jar'
#path_to_models_jar = '../stanford-parser-full-2018-02-27/stanford-chinese-corenlp-2018-02-27-models.jar'
#dependency_parser=StanfordDependencyParser(path_to_jar=path_to_jar,path_to_models_jar=path_to_models_jar,model_path="../stanford-parser-full-2018-02-27/chinesePCFG.ser.gz", encoding="utf-8")

path_to_jar = 'D:/stanford/stanford-parser-full-2018-02-27/stanford-parser.jar'
path_to_models_jar = 'D:/stanford/stanford-parser-full-2018-02-27/stanford-chinese-corenlp-2018-02-27-models.jar'
dependency_parser=StanfordDependencyParser(path_to_jar=path_to_jar,path_to_models_jar=path_to_models_jar,model_path="D:/stanford/stanford-parser-full-2018-02-27/chinesePCFG.ser.gz", encoding="utf-8")


result=dependency_parser.raw_parse(tstr)
dep = result.__next__()
tree=dep.tree()
tree.draw()
for triple in dep.triples():
	if (triple[0][1] in nlist and triple[2][1] in vlist) or (triple[0][1] in vlist and triple[2][1] in nlist) or (triple[0][1] in nlist and triple[2][1] in nlist):
		print(triple)


		
