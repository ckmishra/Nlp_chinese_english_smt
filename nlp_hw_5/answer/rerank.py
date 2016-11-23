#!/usr/bin/env python
import optparse, sys, os, math
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("../data", "test.nbest"), help="N-best file")
optparser.add_option("-w", "--weight-file", dest="weights", default=None, help="Weight filename, or - for stdin (default=use uniform weights)")
optparser.add_option("-e", "--extra_feature", dest="extra_feature", default=os.path.join("./", "test.feature"), help="Extra Feature")

(opts, _) = optparser.parse_args()

w = None
if opts.weights is not None:
  weights_file = sys.stdin if opts.weights is "-" else open(opts.weights)
  w = [float(line.strip()) for line in weights_file]
  w = map(lambda x: 1.0 if math.isnan(x) or x == float("-inf") or x == float("inf") or x == 0.0 else x, w)
  w = None if len(w) == 0 else w

translation = namedtuple("translation", "english, score")
nbests = []

extra_features = []
for n,line in enumerate(open(opts.extra_feature)):
    extra_features.append(line.strip().split("|||")[1])

for n,line in enumerate(open(opts.nbest)):
  (i, sentence, features) = line.strip().split("|||")
  if len(nbests) <= int(i):
    nbests.append([])
  features = [float(h) for h in features.strip().split()]
  new_feature = [float(h) for h in extra_features[n].strip().split()]
  features = features + new_feature
  if w is None or len(w) != len(features):
    w = [1.0/len(features) for _ in xrange(len(features))]
  nbests[int(i)].append(translation(sentence.strip(), sum([x*y for x,y in zip(w, features)])))

for nbest in nbests:
  print sorted(nbest, key=lambda x: -x.score)[0].english
