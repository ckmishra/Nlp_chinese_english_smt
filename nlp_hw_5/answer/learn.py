#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple
from math import fabs
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import bleu
import random,math

translation_candidate = namedtuple("candidate", "sentence, smoothed_scores, features")

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

# untranslated and translated feature
def create_extra_feature(nbest, src, target):
        for n, line in enumerate(open(opts.nbest)):
            (i, sentence, features) = line.strip().split("|||")    
            translated_len_feat = len(sentence.strip().split())
            untranslated_feat = [word for word in sentence.strip().split() if not(is_ascii(word))] 
            yield str(math.log10(len(untranslated_feat)+1))
            #yield  str(translated_len_feat) +" "+str((math.log10(translated_len_feat)-math.log10(len(src))))+" "+str((translated_len_feat-len(src))) +" "+str(len(untranslated_feat))
            #yield  str((math.log10(translated_len_feat))) +" "+ str((len(untranslated_feat)+1))
# get sample
def get_sample(nbest, tau, alpha):
    sample = []
    random.shuffle(nbest);
    for _ in range(tau) :
        s1 = random.choice(nbest)
        s2 = random.choice(nbest)
        if fabs(s1.smoothed_scores - s2.smoothed_scores) > alpha:
            sample.append((s1, s2)) if s1.smoothed_scores > s2.smoothed_scores else sample.append((s2, s1))
        else:
            continue
    return sample

def dot(theta,feat):
    return sum([(x*y) for (x,y) in zip(theta, feat)])

#  perceptron training
def doPerceptron(nbests, epochs, tau, alpha, xi, eta):
    theta = [0.0 for _ in xrange(len(features))]    #initialize theta    
    for i in range(epochs):
        mistakes = 0;
        observed = 0;
        for nbest in nbests:
                samples=get_sample(nbest,tau,alpha);
                for (s1,s2) in sorted(samples, key=lambda (s1,s2): (s1.smoothed_scores - s2.smoothed_scores), reverse=True)[:xi] :
                    if dot(theta, s1.features)  <= dot(theta, s2.features):
                        mistakes += 1
                        diff = [eta*(x-y) for (x,y) in zip(s1.features, s2.features)];
                        theta = [(x+y) for (x,y) in zip(theta, diff)]
                    observed += 1
                    
        sys.stderr.write("\n");
        sys.stderr.write("Mistakes: %d, Error rate: %f, Observerd %d\n" % (mistakes, float(mistakes)/float(observed), observed))          
    print "\n".join([str(weight) for weight in theta])
    
def assign_rank_scores(nbests):
    adjusted = []
    sys.stderr.write("Checking order of bleu and scores...")
    for (k, nbest) in enumerate(nbests):
        current = []
        # sort by descending bleu scores and compare the model scores
        bleu_sorted = sorted(nbest, key=lambda c: -c.smoothed_scores)
        lm_sorted = sorted(nbest, key=lambda c: -sum(c.features))
        for i in xrange(len(bleu_sorted)):
            tmp = copy.deepcopy(lm_sorted[i])
            if bleu_sorted[i] == lm_sorted[i]:
                tmp.features.append(-1.0)
            else:
                tmp.features.append(-1.0 * (math.fabs((i+1) - (bleu_sorted.index(lm_sorted[i]) + 1)) + 1))
            current.append(tmp)
        adjusted.append(current) # maybe shuffle current before appending it?
        if k % 50 == 0:
            sys.stderr.write(".")
    sys.stderr.write("\n")
    return adjusted

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("../data", "train.nbest"), help="N-best file")
    optparser.add_option("-e", "--train_en", dest="train_en", default=os.path.join("../data", "train.en"), help="English source training sentences")
    optparser.add_option("-f", "--train_fr", dest="train_fr", default=os.path.join("../data", "train.fr"), help="French target training sentence")
    optparser.add_option("-t", "--tau", dest="tau",type="int", default=10000, help="Number of samples")
    optparser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.1, help="Alpha value")
    optparser.add_option("-x", "--xi", dest="xi", type="int", default=1000, help="Top x from nbest")
    optparser.add_option("-l", "--eta", dest="eta", type="float", default=0.1, help="Learning rate")
    optparser.add_option("-i", "--epochs", dest="epochs", type="int", default=5, help="Number of iterations")
    optparser.add_option("--af", "--align_score_feature", dest="align_score_feature", default=os.path.join("./", "align.train.feat"), help="Extra Feature")

    (opts, _) = optparser.parse_args()
     
    ref_en = [line.strip().split() for line in open(opts.train_en)]
    
    #alignment feature
    align_features = []
    for n, feat in enumerate(open(opts.align_score_feature)):
        align_features.append(feat)
    # word related features
    wc_features = []
    for n, feat in enumerate(create_extra_feature(opts.nbest,opts.train_fr,opts.train_en)):
        wc_features.append(feat)
    
    nbests = [[] for _ in ref_en]
    for n, line in enumerate(open(opts.nbest)):
        (i, sentence, features) = line.strip().split("|||")
        features = [float(h) for h in features.strip().split()]
        align_feature = [float(h) for h in align_features[n].strip().split()]
        wc_feature = [float(h) for h in wc_features[n].strip().split()]
        features = features + wc_feature +align_feature
        (i, sentence) = (int(i), sentence.strip())
        if len(ref_en) <= i:
          break
        scores = tuple(bleu.bleu_stats(sentence.split(), ref_en[i]))

        smoothed_scores = bleu.smoothed_bleu(scores)
        nbests[i].append(translation_candidate(sentence, smoothed_scores, features))
        if n % 10000 == 0:
          sys.stderr.write(".")
    #nbests = assign_rank_scores(nbests)
    doPerceptron(nbests, opts.epochs, opts.tau, opts.alpha, opts.xi, opts.eta)

    