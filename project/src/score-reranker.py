#!/usr/bin/env python
import optparse, sys, os
import bleu

optparser = optparse.OptionParser()
optparser.add_option("-r", "--reference", dest="reference", default=os.path.join("../test/", "all.cn-en.en"), help="English reference sentences")

(opts,_) = optparser.parse_args()

ref1 = [line.strip().split() for line in open(opts.reference+"0")]
ref2 = [line.strip().split() for line in open(opts.reference+"1")]
ref3 = [line.strip().split() for line in open(opts.reference+"2")]
ref4 = [line.strip().split() for line in open(opts.reference+"3")]

system = [line.strip().split() for line in sys.stdin]

stats = [0 for i in xrange(10)]
for (r1,r2,r3,r4,s) in zip(ref1,ref2,ref3,ref4, system):
  stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats_modified(s,[r1,r2,r3,r4]))]
print bleu.bleu(stats)
