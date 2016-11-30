import optparse, sys, os
from collections import namedtuple
import math
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

if __name__ == '__main__':
    
    optparser = optparse.OptionParser()
    optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("../data", "test.nbest"), help="N-best file")
    optparser.add_option("-r", "--train_en", dest="train_en", default=os.path.join("../data", "test.en"), help="English reference sentences")
    optparser.add_option("-f", "--train_fr", dest="train_fr", default=os.path.join("../data", "test.fr"), help="French Training data")

    (opts, _) = optparser.parse_args()
    
    ref_fr = [line.strip().split() for line in open(opts.train_fr)]

    

        
