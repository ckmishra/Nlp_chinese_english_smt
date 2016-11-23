#!/usr/bin/env python
import optparse, sys, os
from collections import namedtuple
from math import fabs
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import bleu
import random

translation_candidate = namedtuple("candidate", "sentence, smoothed_scores, features")
pair = namedtuple("pair", "s1, s2")

tau = 5000
alpha = 0.1
xi = 100
eta = 0.1
epochs = 5

def get_sample(nbest):
    sample = []
    for time in range(tau) :
        items = random.sample(nbest, 2)
        s1 = items[0]
        s2 = items[1]
        if fabs(s1.smoothed_scores - s2.smoothed_scores) > alpha:
            if s1.smoothed_scores > s2.smoothed_scores:
                sample.append(pair(s1, s2))
            else:
                sample.append(pair(s2, s1))
        else:
            continue
    return sample

if __name__ == '__main__':
    
    optparser = optparse.OptionParser()
    optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("../data", "train.nbest"), help="N-best file")
    optparser.add_option("-r", "--train_en", dest="train_en", default=os.path.join("../data", "train.en"), help="English reference sentences")
    optparser.add_option("-f", "--train_fr", dest="train_fr", default=os.path.join("../data", "train.fr"), help="French Training data")
    optparser.add_option("-e", "--extra_feature", dest="extra_feature", default=os.path.join("./", "extra.feature"), help="Extra Feature")

    (opts, _) = optparser.parse_args()
     
    ref_en = [line.strip().split() for line in open(opts.train_en)]
    extra_features = []
    for n,line in enumerate(open(opts.extra_feature)):
        extra_features.append(line.strip().split("|||")[1])

    nbests = []
    for n, line in enumerate(open(opts.nbest)):
      (i, sentence, features) = line.strip().split("|||")
      features = [float(h) for h in features.strip().split()]
      new_feature = [float(h) for h in extra_features[n].strip().split()]
      features = features + new_feature
      (i, sentence) = (int(i), sentence.strip())
      if len(ref_en) <= i:
        break
      while len(nbests) <= i:
        nbests.append([])
      scores = tuple(bleu.bleu_stats(sentence.split(), ref_en[i]))
      smoothed_scores = bleu.smoothed_bleu(scores)
      nbests[i].append(translation_candidate(sentence, smoothed_scores, features))
      if n%10000==0:
        sys.stderr.write(".")
        
    theta = [0 for _ in xrange(len(features))]

    for i in range(epochs):
        mistakes = 0;
        for nbest in nbests:
            if len(nbest) > 2:
                sample = get_sample(nbest)
                #sort the tau samples from get_sample() using s1.smoothed_bleu - s2.smoothed_bleu
                for (s1,s2) in sorted(sample,key=lambda key: fabs(key.s1.smoothed_scores - key.s2.smoothed_scores))[:xi] :
                #keep the top xi (s1, s2) values from the sorted list of samples
                #do a perceptron update of the parameters theta:
                    if sum([x*y for x,y in zip(theta, s1.features)])  <= sum([x*y for x,y in zip(theta, s2.features)]):
                        mistakes += 1
                        dummy = [eta*(x-y) for x,y in zip(s1.features, s2.features)];
                        theta = [sum(x) for x in zip(theta, dummy)]

        #print "\n" , mistakes           
    print "\n".join([str(weight) for weight in theta])

    