import sys, codecs, optparse, os, math
import pwd
try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('../data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('../data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('../data', '1line'), help="input file to segment")
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
        return '{}: {} {}'.format(self.__class__.__name__,
                                  self.word,
                                  self.startIndex)
    
def is_in_queue(x, q):
   with q.mutex:
      return x in q.queue

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
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
        self.missingfn = missingfn or (lambda k, N: 1./N)

    def __call__(self, key):
        if key in self: return math.log(float(self[key])/float(self.N))
        elif len(key) == 1 : return math.log(self.missingfn(key, self.N))
        else : return None 
            #return math.log(1./(self.N * 10**len(key)))


# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)
old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# ignoring the dictionary provided in opts.counts
pq = Q.PriorityQueue()
with open(opts.input) as f:
    for line in f:
        utf8line = unicode(line.strip(), 'utf-8')  
        input = [i for i in utf8line]  # segmentation is one word per character in the input
        chart = {}
        for j in range(1,min(1 + Pw.maxlen,len(input))+1):
            word = "".join(input[:j])
            if Pw(word) is None:
                    continue
            else:
                pq.put(Entry(word, 0, Pw(word), None));

        ## Iteratively fill in chart[i] for all i ##
        while not pq.empty() :
             entry = pq.get();
             endIndex =  entry.startIndex + len(entry.word)
            
             if endIndex in chart :
                prevEntry = chart[endIndex];
                if entry.logProb > prevEntry.logProb :
                    chart[endIndex] = entry;
                else:
                    continue
             else :
                 chart[endIndex] = entry
            
             for k in range(endIndex+1, min(endIndex + 1 + Pw.maxlen, len(input))+1): 
                newWord = "".join(input[endIndex:k])  
                if Pw(newWord) is None:
                    continue
                else : 
                    newEntry = Entry(newWord, endIndex, entry.logProb + Pw(newWord), entry)
                    if(not(is_in_queue(newEntry,pq))):
                        pq.put(newEntry);   
            
        ## Get the best segmentation ##
        finalIndex = len(input);
        finalentry = chart[finalIndex]
        result = [];
        while finalentry is not None:
            result.insert(0,finalentry.word);
            finalentry = finalentry.backptr
        print " ".join(result)

sys.stdout = old