import sys
import model

from zweeBasic import *

h = model.train()
for i in h.history:
    print(i)
    
    if i == "loss":
        TestLang.LOSS = h.history[i][-1]
    elif i == "val_loss":
        TestLang.VAL_LOSS = h.history[i][-1]
    elif i == "acc":
        TestLang.ACC = h.history[i][-1]
    elif i == "val_acc":
        TestLang.VAL_ACC = h.history[i][-1]
    elif i == "precision":
        TestLang.PRECISION = h.history[i][-1]
    elif i == "val_precision":
        TestLang.VAL_PRECISION = h.history[i][-1]
    elif i == "recall":
        TestLang.RECALL = h.history[i][-1]
    elif i == "val_recall":
        TestLang.VAL_RECALL = h.history[i][-1]
    
    print(h.history[i])
    import numpy
    print(type(h.history[i][-1]) == numpy.float32)
    print("length = " + str(len(h.history[i]))) 
    print()

model.predict()
from time import time
from pymongo import MongoClient
client = MongoClient('3.94.109.234')
db = client.ci

build_info = {"build_time" : int(time())}
def _convert(t):
    if type(t) == numpy.float32:
        return float(t)
    return t
build_info['history'] = {} 
for i in h.history:
    build_info['history'][i] = []
    for j in h.history[i]:
        build_info['history'][i].append(_convert(j))
for param in dir(TestLang):
    if param.isupper():
        build_info[param] = _convert(getattr(TestLang, param))
print(build_info) 

print("Prev build")
import pymongo
prev_build = list(db.builds.find().sort([('build_time', pymongo.DESCENDING)]))[0]
print(prev_build)
TestLang.prev = TestLangClass()
for k, v in prev_build.items():
    if k.isupper():
        setattr(TestLang.prev, k, v)

db.builds.insert(build_info)
try:
    import tests
except:
    print("TESTS FAILED")
    sys.exit(-1)
