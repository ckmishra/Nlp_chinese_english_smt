#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
import math
import pickle
#from datashape.typesets import maxtype

optparser = optparse.OptionParser()
#basedir=".."
basedir = "/usr/shared/CMPT/nlp-class/project";
optparser.add_option("-e", "--english", dest="english", default=basedir+"/medium/train.en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default=basedir+"/medium/train.cn", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("--nb", "--nbest", dest="nbest", default=os.path.join("./", "nbest.txt"), help="N-best file")
optparser.add_option("--tf", "--train_fr", dest="train_fr", default=os.path.join("../toy/", "train.cn"), help="French Training data")
(opts, _) = optparser.parse_args()
'''
f_data = "%s" % (opts.french)
e_data = "%s" % (opts.english)
if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)
    
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]


def model(f_data,e_data):
    sys.stderr.write("Training with EM Method ...")
    f_count = defaultdict(int)
    e_count = defaultdict(int)
    fe_count = defaultdict(int)
    for (n, (f, e)) in enumerate(bitext):
        e = e.insert(0,None);
        
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            f_count[f_i] += 1
            for e_j in set(e):
                fe_count[(f_i,e_j)] += 1

    sys.stderr.write("\n")
    
         
    t = defaultdict(float)
    for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
        t[(f_i,e_j)] = 1 / float(len(f_count))
    
    for i in range(15):
        sys.stderr.write("Iteration %d " %i)
        t_e_count = defaultdict(float)
        t_fe_count = defaultdict(float)
        for (n, (f, e)) in enumerate(bitext):
            for f_i in set(f):
                Z=0.0
                for e_j in set(e):
                    Z+=t[(f_i,e_j)]
                for e_j in set(e):
                    c = t[(f_i,e_j)]/Z
                    t_fe_count[(f_i,e_j)]+=c
                    t_e_count[(e_j)]+=c
            if n % 5000 == 0:
                sys.stderr.write(".")
        sys.stderr.write("\n")        
                
        for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
            t[(f_i,e_j)] = (t_fe_count[(f_i,e_j)]+.005) / (t_e_count[(e_j)]+(.005*100000))
            if k % 5000 == 0:
                sys.stderr.write("%")
        sys.stderr.write("\n")         

    return t

t1 = model(f_data, e_data)

with open('IBMModel1_cn_en.pickle', 'wb') as handle:
    pickle.dump(t1, handle, protocol=pickle.HIGHEST_PROTOCOL)

print "Pickling Done.."

'''
t1 = {}
with open('../IBMModel1_cn_en.pickle', 'rb') as handle:
    t1 = pickle.load(handle)

ref_fr = [line.strip().split() for line in open(opts.train_fr[:opts.num_sents])]

token = 0

for i,f in enumerate(ref_fr):
        token += len(f)

smoothValue = 1/float(token) 

for n,line in enumerate(open(opts.nbest)):
    (i, sentence, features) = line.strip().split("|||")
    (i, sentence) = (int(i), sentence.strip())
    finalscore = 0
    for (i, f_i) in enumerate(ref_fr[i]):
        score = 0
        for (j, e_j) in enumerate(sentence.split()):
            score += t1.get((f_i,e_j), smoothValue)
        finalscore +=  math.log10(score)    
    print finalscore

