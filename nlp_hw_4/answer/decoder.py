#!/usr/bin/env python
import optparse, sys, itertools, copy, math, time, os, gzip
from collections import namedtuple, defaultdict
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import models

hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage, end, logprob_with_fc")
entry = namedtuple("entry", "start, end, phrases") # used in get_all_phrases for keeping triple 

def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)

# tm should translate unknown words as-is with probability 1
def handle_unk_words(french, tm):
    for word in set(sum(french,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]

def get_all_phrases(h, f, tm, f_len, distortion_limit):
    all_phrases = []
    untranslated = (x for (x, v) in enumerate(h.coverage) if v == 0)
    for start in untranslated:
        #if abs(h.end - start) <= distortion_limit or f_len > 20 : # limiting distortion, for larger sentence (>20) relaxing distortion
            for end in xrange(start+1, f_len+1):
                translated = h.coverage[start:end].count(1)
                if translated == 0:
                    words = f[start:end]
                    if words in tm:
                        all_phrases.append(entry(start, end, tm[words]))
    return  all_phrases   
         
# Future cost table created using dynamic programming
def pre_compute_future_cost(f, tm, lm, f_len):
    costs = defaultdict(float)
    for length in xrange(1, f_len+1):
        for start in xrange(0 , f_len + 1 - length):
            end = start + length
            costs[(start,end)] =float("-inf")
            phrases = tm.get(f[start:end],[])
            for phrase in phrases:
                logprob = phrase.logprob
                lm_state = tuple()
                for word in phrase.english.split():
                    (lm_state, word_logprob) = lm.score(lm_state,word)
                    logprob += word_logprob
                costs[(start,end)] = max(costs[(start,end)], logprob) 
            for i in xrange(start+1,end):
                   costs[(start,end)] = max(costs[(start,end)] , costs[(start,i)] + costs[(i,end)])
    return costs

# get value from future cost table
def get_future_cost(coverage, f_costs, f_len):
    future_logprob = 0.0
    start = 0; # finding index having first zero  
    found = False # flag for getting sequence of zeros
    for i in xrange(f_len):
        if not found and coverage[i] == 0:
            start = i   
            found = True
        elif found and coverage[i] == 1 :
            future_logprob += f_costs[(start, i)]
            found = False # reset found
        elif found and (i == f_len-1):
            # reached end of sentence
            future_logprob += f_costs[(start, f_len)]
    return future_logprob
 
                                 
def decode(french, tm, lm, stack_max, distortion_limit, distortion_penalty, verbose, beam_width):
    # lambda to learn, can give weight to different features
    lambda_tm = 1
    lambda_d  = 1
    lambda_lm = 1

    sys.stderr.write("Decoding %s...\n" % opts.input)
    for (s, f) in enumerate(french):
        f_len = len(f)
        coverage = [0 for _ in f] # init coverage 
        initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, coverage, 0, 0.0) # initial hypothesis
        stacks = [{} for _ in xrange(f_len + 1)]
        stacks[0][(0, lm.begin(), tuple(coverage))] = initial_hypothesis
        f_costs = pre_compute_future_cost(f, tm, lm, f_len) # get the future cost table using dynamic programming

        for i, stack in enumerate(stacks[:-1]):
            #maxValue = max(stack.itervalues(), key=lambda h: h.logprob_with_fc).adjusted_logprob
            #for h in sorted(item for item in stack.itervalues() if item.logprob_with_fc >= (maxValue-beam_width)): # using beam width
            for h in sorted(stack.itervalues(),key=lambda h: -h.logprob_with_fc)[:stack_max]:
                for entry in get_all_phrases(h, f, tm, f_len, distortion_limit):
                    coverage = copy.deepcopy(h.coverage) 
                    for c in xrange(entry.start, entry.end):
                        coverage[c] = 1 # updated coverage vector
                    covered = coverage.count(1)
                    
                    for phrase in entry.phrases:
                        # translation model
                        logprob = h.logprob + (lambda_tm * phrase.logprob) 
                        lm_state = h.lm_state
                        # language model
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            logprob +=  (lambda_lm * word_logprob)
                        logprob += (lambda_lm*lm.end(lm_state)) if covered == f_len else 0.0 
                        # future cost
                        future_logprob = get_future_cost(coverage, f_costs, f_len)
                        # giving distortion penalty
                        penalty =  abs(entry.start - h.end) * math.log10(distortion_penalty)
                        logprob_with_fc = future_logprob + lambda_d*penalty + logprob  # updated logprob
                       
                        new_hypothesis = hypothesis(logprob, lm_state, h, phrase, coverage, entry.end, logprob_with_fc)
                        key = (lm_state, tuple(coverage))
                        if key not in stacks[covered] or stacks[covered][key].logprob_with_fc < logprob_with_fc:
                            stacks[covered][key] = new_hypothesis
        
        assert(len(stacks[-1]) > 0)
        # get top and backtrack
        top = max(stacks[-1].itervalues(), key=lambda h: h.logprob_with_fc)
        assert(top.coverage.count(1) == len(top.coverage))
        print extract_english(top)
        if verbose:
            tm_logprob = extract_tm_logprob(top)
            lm_logprob = top.logprob - tm_logprob
            sys.stderr.write("s = %d, LM = %f, TM = %f, Total = %f\n" % (s, lm_logprob, tm_logprob,top.logprob_with_fc))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="../data/input", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="../data/tm", help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="../data/lm", help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=20, type="int", help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=100, type="int", help="top stack size (default=1)")
    optparser.add_option("-d", "--distortion-limit", dest="d", default=10, type="int", help="top distortion limit (default=1)")
    optparser.add_option("-p", "--distortion-penalty", dest="p", default=0.9, type="float", help="Distortion penalty (default=0.9)")
    optparser.add_option("-b", "--beam-width", dest="b", default=2, type="int", help="Beam width parameter (default=10)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
    optparser.add_option("-o", "--logfile", dest="logfile", default=None, help="filename for logging output")

    opts = optparser.parse_args()[0]
    
    start_time = time.time()

    tm = models.TM(opts.tm, opts.k)
    lm = models.LM(opts.lm)
    french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
    handle_unk_words(french, tm)
    decode(french, tm, lm, opts.s, opts.d, opts.p, opts.verbose,opts.b)
    sys.stderr.write("\nTook %s seconds\n" % (time.time() - start_time))
