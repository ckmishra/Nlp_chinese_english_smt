import sys, codecs, optparse, os, math, re
import pwd
try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('../data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('../data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('../data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()


class Entry(object):
    def __init__(self, word, startIndex, logProb, backptr):
        self.word = word
        self.startIndex = startIndex
        self.logProb = logProb
        self.backptr = backptr
        
    def __eq__(self, other):
        return self.word == other.word and self.startIndex == other.startIndex
    
    def __cmp__(self, other):
        return cmp(other.logProb, self.logProb)

    def __str__(self, *args, **kwargs):
        return self.word
    
    def __repr__(self):
        return '{}: {} {} {} {}'.format(self.__class__.__name__,
                                  self.word,
                                  self.startIndex, self.logProb, self.backptr)
    
def is_in_queue(x, q):
   with q.mutex:
      return x in q.queue
  
def avoid_long_words(key, N):
    "Estimate the probability of an unknown word."
    return 10000. / (N * 10000 ** len(key))

digitRegex = re.compile(ur"^\d+\Z", re.UNICODE)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=avoid_long_words):
        self.maxlen = 0 
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1. / N)

    def __call__(self, key):
        if key in self: return float(self[key]) / float(self.N)
        elif digitRegex.match(key): return 0.1
        else : return self.missingfn(key, self.N)
     
    
class PdistJoint(dict):
    "A probability distribution estimated from counts in datafile."
    """ P(x,y) joint prob only """

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            joint_tuple, freq = line.split(sep)
            try:
                utf8key = unicode(joint_tuple, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1. / N)

    def __call__(self, key):
        if key in self:
            return float(self[key]) / float(self.N)
        else :
            return None
        
        
class PdistUnigram(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=avoid_long_words):
        self.maxlen = 0
        for line in file(filename):
            tuple, freq = line.split(sep)
            (tuple_1, tuple_2) = tuple.strip().split(" ")
            try:
                utf8key_1 = unicode(tuple_1, 'utf-8')
                utf8key_2 = unicode(tuple_2, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key_1] = self.get(utf8key_1, 0) + int(freq)
            self[utf8key_2] = self.get(utf8key_2, 0) + int(freq)
            self.maxlen = max(max(len(utf8key_1), self.maxlen), len(utf8key_2))
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1. / N)
    
    def __call__(self, key):
        if key in self: return float(self[key]) / float(self.N)
        elif digitRegex.match(key): return 0.1
        else : return self.missingfn(key, self.N)

# the default segmenter does not use any probabilities, but you could ...
# Pw  = Pdist(opts.counts1w)
Pw = PdistUnigram(opts.counts2w)
PwJoint = PdistJoint(opts.counts2w)
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
pq = Q.PriorityQueue()
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')  
        input = [i for i in utf8line]
        chart = {}
        for j in range(1, min(1 + Pw.maxlen, len(input)) + 1):
            newWord = "".join(input[:j])
            prev_word = unicode("<S>", 'utf-8');
            joint_tuple = " ".join([prev_word, newWord]);
            joint_prob = PwJoint(joint_tuple)
            
            unigramPrevProb = Pw(prev_word)

            if  joint_prob is None:
                # print "Inside"
                unigramProb = Pw(newWord)
                if unigramProb is not None:
                    newEntry = Entry(newWord, 0, math.log10(unigramProb), None)
                    if(not(is_in_queue(newEntry, pq))):
                       pq.put(newEntry);   
            else:
                newEntry = Entry(newWord, 0, math.log10(joint_prob) - math.log10(unigramPrevProb), None)
                if(not(is_in_queue(newEntry, pq))):
                    pq.put(newEntry);
        
        # # Iteratively fill in chart[i] for all i ##
        while not pq.empty() :
            entry = pq.get();
            endIndex = entry.startIndex + len(entry.word) - 1;
            
            if endIndex in chart :
                prevEntry = chart[endIndex];
                if entry.logProb > prevEntry.logProb :
                    chart[endIndex] = entry;
                else:
                    continue
            else :
                 chart[endIndex] = entry
            
            for k in range(endIndex, min(endIndex + Pw.maxlen, len(input)) + 1): 
                nextword = "".join(input[endIndex + 1:k]) 
                
                entry = chart[endIndex];
                prev_word = entry.word
                joint_tuple = " ".join([prev_word, nextword]);
                joint_prob = PwJoint(joint_tuple)

                unigramPrevProb = Pw(prev_word)

                if joint_prob is None:
                    # print "Inside"
                    # continue
                    unigramProb = Pw(nextword)
                    if unigramProb is not None:
                        newEntry = Entry(nextword, endIndex + 1, entry.logProb + math.log10(unigramProb), endIndex)
                        if(not(is_in_queue(newEntry, pq))):
                            pq.put(newEntry);
                else :
                   newEntry = Entry(nextword, endIndex + 1, entry.logProb + math.log10(joint_prob) - math.log10(unigramPrevProb), endIndex)
                   if(not(is_in_queue(newEntry, pq))):
                        pq.put(newEntry);   
            
        # # Get the best segmentation ##
        finalIndex = len(input) - 1;
        finalentry = chart[finalIndex]
        result = [];
        
        while True:
            result.insert(0, finalentry.word);
            if finalentry.backptr is not None :
                finalentry = chart[finalentry.backptr]
            else :
                break
                
        print " ".join(result)
        
sys.stdout = old
