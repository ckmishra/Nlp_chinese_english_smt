#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple
from math import fabs
from docutils.nodes import thead
import bleu
import random,math

translation_candidate = namedtuple("candidate", "sentence, smoothed_scores, features, rank")
 
# get sample
def get_sample(nbest, tau, alpha):
    sample = []
    random.shuffle(nbest);
    for _ in range(tau) :
        s1 = random.choice(nbest)
        s2 = random.choice(nbest)
        if abs(s1.rank - s2.rank) > 10 and ((s1.rank/float(s2.rank)) > 2 or (s2.rank/float(s1.rank)) > 2 ) and fabs(s1.smoothed_scores - s2.smoothed_scores) > alpha:
            if s1.smoothed_scores > s2.smoothed_scores and (s1,s2) not in sample:
                sample.append((s1, s2)) 
            elif s2.smoothed_scores > s1.smoothed_scores and (s2,s1) not in sample:
                sample.append((s2, s1))
        else:
            continue
    return sample

def dot(theta,feat):
    return sum([(x*y) for (x,y) in zip(theta, feat)])

#  perceptron training
def doPerceptron(nbests,feature_len, epochs, tau, alpha, xi, eta, theta):
    for i in range(epochs):
        mistakes = 0;
        observed = 0; # to avoid divide by Zero
        for nbest in nbests:
                samples=get_sample(nbest,tau,alpha);
                sorted_sample = sorted(samples, key=lambda (s1,s2): (s1.smoothed_scores - s2.smoothed_scores), reverse=True)[:xi];
                for (s1,s2) in sorted_sample :
                    if dot(theta, s1.features)  <= dot(theta, s2.features):
                        mistakes += 1
                        diff = [eta*(x-y) for (x,y) in zip(s1.features, s2.features)];
                        theta = [(x+y) for (x,y) in zip(theta, diff)]                       
                        try:
                            nbest.remove(s1)
                            nbest.remove(s2)
                        except:
                            pass
                    observed += 1
        sys.stderr.write("\n");
        sys.stderr.write("Mistakes: %d, Error rate: %f, Observerd %d\n" % (mistakes, float(mistakes)/float(observed), observed))          
    
    print "\n".join([str(weight) for weight in theta])

def assign_rank(nbests):
    updates_nbests = [[] for _ in nbests]
    for (k, nbest) in enumerate(nbests):
        nbest = sorted(nbest, key= lambda elem : - elem.smoothed_scores)
        for i,candidate in enumerate(nbest):
                candidate = translation_candidate(candidate.sentence, candidate.smoothed_scores, candidate.features, i+1)
                updates_nbests[k].append(candidate) 
    return updates_nbests

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    #basedir ="/usr/shared/CMPT/nlp-class/project"
    basedir=".."
    optparser.add_option("-i", "--nbest", dest="nbest", default="./nbest.txt", help="N-best file")
    optparser.add_option("-f", "--train_fr", dest="train_fr", default=os.path.join(basedir+"/dev/all.cn-en.cn"), help="French target training sentence")
    optparser.add_option("--e1", "--train_en1", dest="train_en1", default=os.path.join(basedir+"/dev/all.cn-en.en0"), help="English ref  sentences")
    optparser.add_option("--e2", "--train_en2", dest="train_en2", default=os.path.join(basedir+"/dev/all.cn-en.en1"), help="English ref training sentences")
    optparser.add_option("--e3", "--train_en3", dest="train_en3", default=os.path.join(basedir+"/dev/all.cn-en.en2"), help="English ref training sentences")
    optparser.add_option("--e4", "--train_en4", dest="train_en4", default=os.path.join(basedir+"/dev/all.cn-en.en3"), help="English ref  sentences")
    optparser.add_option("--num", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default= no limit)")
    optparser.add_option("--th", "--theta", dest="theta", default=None, help="Theta value")

    optparser.add_option("-t", "--tau", dest="tau",type="int", default=5000, help="Number of samples")
    optparser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.001, help="Alpha value")
    optparser.add_option("-x", "--xi", dest="xi", type="int", default=100, help="Top x from nbest")
    optparser.add_option("-l", "--eta", dest="eta", type="float", default=0.1, help="Learning rate")
    optparser.add_option("-n", "--epochs", dest="epochs", type="int", default=5, help="Number of iterations")
    #optparser.add_option("--af", "--align_score_feature", dest="align_score_feature", default=os.path.join("./", "align.train.feat"), help="Extra Feature")
    #optparser.add_option("-o", "--output", dest="output", type="int", default=os.path.join(basdir+ "train.cn"), help=" Learned weight")

    (opts, _) = optparser.parse_args()
     
    ref_en1 = [line.strip().split() for line in open(opts.train_en1)]
    ref_en2 = [line.strip().split() for line in open(opts.train_en2)]
    ref_en3 = [line.strip().split() for line in open(opts.train_en3)]
    ref_en4 = [line.strip().split() for line in open(opts.train_en4)]
    
    src = [line.strip().split() for line in open(opts.train_fr)]
    

    nbests = [[] for _ in ref_en1[:int(opts.num_sents)]]# take only specified sentence

    #sys.stderr.write(opts.nbest)
    #sys.stderr.write(opts.train_en1)
    for n, line in enumerate(open(opts.nbest,'r')):    
        (i, sentence, features) = line.strip().split("|||")
        features = [float(h) for h in features.strip().split()]
        features = features #   + align_feature + wc_feature
        (i, sentence) = (int(i), sentence.strip())
        if len(ref_en1) <= i:
          break
        # using my own bleu
        scores = tuple(bleu.bleu_stats_modified(sentence.split(),[ref_en1[i],ref_en2[i],ref_en3[i], ref_en4[i]]))
        smoothed_scores = bleu.smoothed_bleu(scores)   

        nbests[i].append(translation_candidate(sentence, smoothed_scores, features, 0))
        if n % 10000 == 0:
          sys.stderr.write(".")
          sys.stderr.write("Smoothed Score...%f" % smoothed_scores) 
          
    nbests = assign_rank(nbests)# ranking
    sys.stderr.write("Ranking Completed ..." )    
    sys.stderr.write("Percpeptron Started %d" % len(nbests))
    
    if opts.theta is None :
        theta = [0.0 for _ in xrange(len(features))]    #initialize theta  
    else :
        theta = [float(line) for line in open(opts.theta,'r')] 
    doPerceptron(nbests, len(features), opts.epochs, opts.tau, opts.alpha, opts.xi, opts.eta, theta)
