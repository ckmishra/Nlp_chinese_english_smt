#!/usr/bin/env python
import optparse, sys, os, math
from collections import namedtuple
import feature_util

optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("../data", "test.nbest"), help="N-best file")
optparser.add_option("-f", "--test_fr", dest="test_fr", default=os.path.join("../data", "test.fr"), help="French Training data")
optparser.add_option("-w", "--weight-file", dest="weights", default=None, help="Weight filename, or - for stdin (default=use uniform weights)")
optparser.add_option("--af", "--align_score_feature", dest="align_score_feature", default=os.path.join("./", "align.test.feat"), help="Alignment score Feature")

(opts, _) = optparser.parse_args()

    
def create_word_features(nbest, src):
    for n, line in enumerate(open(opts.nbest)):
        (i, sentence, features) = line.strip().split("|||")    
        translated_len_feat = len(sentence.strip().split())
        untranslated_feat = [word for word in sentence.strip().split() if not(feature_util.is_ascii(word))] 
        yield  str(math.log10(translated_len_feat)) +" "+ str((len(untranslated_feat))) +" "+str(feature_util.quotationMatch(sentence))

w = None
if opts.weights is not None:
  weights_file = sys.stdin if opts.weights is "-" else open(opts.weights)
  w = [float(line.strip()) for line in weights_file]
  w = map(lambda x: 1.0 if math.isnan(x) or x == float("-inf") or x == float("inf") or x == 0.0 else x, w)
  w = None if len(w) == 0 else w

translation = namedtuple("translation", "english, score")
nbests = []

src = [line.strip().split() for line in open(opts.test_fr)]
# IBM model 1 score feature for test data
align_features = []
for n, feat in enumerate(open(opts.align_score_feature)):
    align_features.append(feat)

# word count specific feature    
wc_features = []
for n, feat in enumerate(create_word_features(opts.nbest, src)):
    wc_features.append(feat)

for n,line in enumerate(open(opts.nbest)):
  (i, sentence, features) = line.strip().split("|||")
  if len(nbests) <= int(i):
    nbests.append([])
  features = [float(h) for h in features.strip().split()]
  align_feature = [float(h) for h in align_features[n].strip().split()]
  wc_feat = [float(h) for h in wc_features[n].strip().split()]
  features = features # + align_feature + wc_feat

  if w is None or len(w) != len(features):
    w = [1.0/len(features) for _ in xrange(len(features))]
  nbests[int(i)].append(translation(sentence.strip(), sum([x*y for x,y in zip(w, features)])))

for nbest in nbests:
  print sorted(nbest, key=lambda x: -x.score)[0].english
