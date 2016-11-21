#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict
from datashape.typesets import maxtype

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
bitext1 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
for (n, (f, e)) in enumerate(bitext1):
        e.insert(0,None);
        
bitext2 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(e_data), open(f_data))[:opts.num_sents]]
for (n, (f, e)) in enumerate(bitext2):
        e = e.insert(0,None);
        
def model(f_data,e_data):

    sys.stderr.write("Training with EM Method ...")
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    f_count = defaultdict(int)
    e_count = defaultdict(int)
    fe_count = defaultdict(int)
    #t_e_count = defaultdict(float)
    #t_fe_count = defaultdict(float)
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
    
    for i in range(6):
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
            t[(f_i,e_j)] = (t_fe_count[(f_i,e_j)]+.005) / (t_e_count[(e_j)]+500)
            if k % 5000 == 0:
                sys.stderr.write("%")
        sys.stderr.write("\n")         

    return t

t1 = model(f_data, e_data)
t2 = model(e_data, f_data)


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
                if maxj is not 0:        
                    Li.append("%i-%i " % (i,maxj-1))      
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
                if maxj is not 0:        
                    Li.append("%i-%i " % (maxj-1,i))  
            L2.append(Li)   
            
            
for (n, l) in enumerate(L1):
    for (i, l_i) in enumerate(l):
        if l_i in L2[n]:
            sys.stdout.write(l_i)
    sys.stdout.write("\n")   
    
    

    

