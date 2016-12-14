#!/usr/bin/env python
# Simple translation model and language model data structures
import sys
from collections import namedtuple
import gzip

def extract_data(filepath):
    return gzip.open(filepath, 'r') if filepath[-3:] == '.gz' else open(filepath, 'r')
# A translation model is a dictionary where keys are tuples of French words
# and values are lists of (english, logprob) named tuples. For instance,
# the French phrase "que se est" has two translations, represented like so:
# tm[('que', 'se', 'est')] = [
#   phrase(english='what has', logprob=-0.301030009985), 
#   phrase(english='what has been', logprob=-0.301030009985)]
# k is a pruning parameter: only the top k translations are kept for each f.
phrase = namedtuple("phrase", "english, features, logprob")
def TM(filename, k, weights=[1,1,1,1]):
  sys.stderr.write("Reading translation model from %s...\n" % (filename,))
  tm = {}
  for line in extract_data(filename):
    (f, e, features) = line.strip().split(" ||| ")
    # handling four features
    features = features.strip().split()
    features = [float(feat) for feat in features]
    weightedLogProb=0.0
    for (feat,w) in zip(features,weights):
        weightedLogProb += (feat*w)
    tm.setdefault(tuple(f.split()), []).append(phrase(e, features, weightedLogProb) )
  for f in tm: # prune all but top k translations
    tm[f].sort(key=lambda x: -x.logprob)
    del tm[f][k:] 
  return tm

# # A language model scores sequences of English words, and must account
# # for both beginning and end of each sequence. Example API usage:
# lm = models.LM(filename)
# sentence = "This is a test ."
# lm_state = lm.begin() # initial state is always <s>
# logprob = 0.0
# for word in sentence.split():
#   (lm_state, word_logprob) = lm.score(lm_state, word)
#   logprob += word_logprob
# logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
ngram_stats = namedtuple("ngram_stats", "logprob, backoff")



class LM:
  def __init__(self, filename):
    sys.stderr.write("Reading language model from %s...\n" % (filename,))
    self.table = {("<unk>",):ngram_stats(-5.369621,0.0)} # unk word
    for line in extract_data(filename):
      entry = line.strip().split("\t")
      if len(entry) > 1 and entry[0] != "ngram":
        (logprob, ngram, backoff) = (float(entry[0]), tuple(entry[1].split()), float(entry[2] if len(entry)==3 else 0.0))
        self.table[ngram] = ngram_stats(logprob, backoff)

  def begin(self):
    return ("<s>",)

  def score(self, state, word):
    ngram = state + (word,)
    score = 0.0
    while len(ngram)> 0:
      if ngram in self.table:
        return (ngram[-2:], score + self.table[ngram].logprob)
      else: #backoff
        score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0 
        ngram = ngram[1:]
    return ((), score + self.table[("<unk>",)].logprob)
    
  def end(self, state):
    return self.score(state, "</s>")[1]
