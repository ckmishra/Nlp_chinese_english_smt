#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with EM ...\n")

bitext1 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

bitext2 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))[:opts.num_sents]]

def model(f_data,e_data):
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    k = 0
    # initialize uniformly
    #tPara = defaultdict(int)
    f_count = defaultdict(int)
   
    for (f, e) in bitext:
        for f_i in enumerate(f):
            f_count[f_i] +=1
    sys.stderr.write("Initialization complete ...\n")
    initValue = 1/float(len(f_count));
    # EM        
    sys.stderr.write("Training IBM Model 1 (with nulls) with Expectation Maximization...")
    for iter in range(10):
        sys.stderr.write("\nIteration %d" %iter)
        k+=1
        e_count = defaultdict(int)
        fe_count = defaultdict(int)
        for (n,(f, e)) in enumerate(bitext):
          for f_i in set(f):
            Z  = 0
            for e_j in set(e) :
                if k-1 == 0:
                    Z +=initValue;
                else:
                    Z += tPara[(f_i,e_j)]
            for e_j in set(e) :
                if k-1 == 0:
                    c = initValue/float(Z)
                else :
                    c = tPara[(f_i,e_j)]/Z
                fe_count[(f_i,e_j)] += c
                e_count[e_j] += c
          if n % 5000 == 0:
              sys.stderr.write(".")
    
        tPara = defaultdict(float)                 
        for (n, (f_i, e_j)) in enumerate(fe_count.keys()):
             tPara[(f_i, e_j)] = (fe_count[(f_i, e_j)]+ 0.01)/float(e_count[e_j]+ (0.01*10000))
    
    return tPara





t1 = model(f_data, e_data)
t2 = model(e_data, f_data)

sys.stderr.write("\naligning....")

L1 = [] 
for (n, (f, e)) in enumerate(bitext1):
            Li = []
            for (i, f_i) in enumerate(f):
                maxt = 0
                maxj = 0
                for (j, e_j) in enumerate(e):
                    if t1[(f_i,e_j)]> maxt:
                        maxt =t1[(f_i,e_j)]
                        maxj=j
                Li.append("%i-%i " % (i,maxj))
            L1.append(Li)    
            
L2 = []
for (n, (f, e)) in enumerate(bitext2):
            Li = []
            for (i, f_i) in enumerate(f):
                maxt = 0
                maxj = 0
                for (j, e_j) in enumerate(e):
                    if t2[(f_i,e_j)]> maxt:
                        maxt =t2[(f_i,e_j)]
                        maxj=j
                Li.append("%i-%i " % (maxj,i))
            L2.append(Li)   
            
            
for (n, l) in enumerate(L1):
    for (i, l_i) in enumerate(l):
        if l_i in L2[n]:
            sys.stdout.write(l_i)
    sys.stdout.write("\n")   

