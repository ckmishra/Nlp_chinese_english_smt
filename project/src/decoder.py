#!/usr/bin/env python
import optparse, sys, itertools, copy, math, time, os, gzip
from collections import namedtuple, defaultdict
from sys import stdin, stdout
import models
import subprocess
from subprocess import  PIPE, Popen
import re
import pickle

hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, coverage, end, logprob_with_fc, features")
entry = namedtuple("entry", "start, end, phrases") # used in get_all_phrases for keeping triple 

feat_len = 7;

def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)

def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)


# Checking is hypothesis valid 
def isValidDistrotion(coverage,distortion):
    prefix_one_bit=0; # counting number of 1 before first zero
    for i in coverage:
        if i==0:
            break;
        prefix_one_bit += 1
    
    last_one_bit=0; # getting index of last 1
    for i,j in enumerate(coverage):
        if j==1:
            last_one_bit = i;
    
    if(last_one_bit - prefix_one_bit) <= distortion:# if condition true then allow
        return True

    return False

#tm should translate unknown words as-is with probability 1
def handle_unk_words(french, tm):
    for word in set(sum(french,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, [0.0,0.0,0.0,0.0], 0.0)]

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
 
                                 
def decode(french, tm, lm, ibm_model, stack_max, distortion_limit, distortion_penalty, verbose, beam_width, weights, nbest_size):
    result = [];

    sys.stderr.write("Decoding %s...\n" % opts.input)
    for (s, f) in enumerate(french):
        if(s % 100)==0:
            sys.stderr.write(".")
        f_len = len(f)
        coverage = [0 for _ in f] # init coverage 
        initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, coverage, 0, 0.0, [0.0 for _ in range(len(weights))]) # initial hypothesis
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
                        if distortion_limit != -1 and not isValidDistrotion(coverage, distortion_limit): 
                            # if distortion limit -1 then no limit on distortion
                                continue
                        # translation model
                        tm_logProbs = [feat*weight for (feat,weight) in zip(phrase.features, weights[2:6])]
                        tm_logProb = sum(tm_logProbs)
                        
                        logprob = h.logprob + tm_logProb
                        lm_state = h.lm_state
                        # language model
                        lm_logprob = 0.0;
                        for word in phrase.english.split():
                            (lm_state, word_logprob) = lm.score(lm_state, word)
                            lm_logprob +=  ( word_logprob)
                        lm_logprob += (lm.end(lm_state)) if covered == f_len else 0.0 
                        logprob += weights[0]*lm_logprob;
                
                        # IBM 1 model score
                        align_logProb = 0.0;
                        for f_i in f[entry.start:entry.end]:
                            score = 0
                            for e_j in phrase.english.split():
                                score+= ibm_model.get((f_i,e_j), 1/float(1000000))#if not found then smoothing
                            align_logProb +=  math.log10(score) 
                        logprob += weights[6]*align_logProb;
                
                
                        #giving distortion penalty
                        penalty =  abs(entry.start - h.end) * math.log10(distortion_penalty)
                        logprob += (weights[1] * penalty) 
                        # future cost
                        future_cost = get_future_cost(coverage, f_costs, f_len)
                        logprob_with_fc = future_cost + logprob  # updated logprob
                        
                        features = [lm_logprob, penalty] + phrase.features +[align_logProb] # hypothesis features
                        hyp_features = [x+y for (x,y) in zip(features,h.features)]
                        
                        new_hypothesis = hypothesis(logprob, lm_state, h, phrase, coverage, entry.end, logprob_with_fc, hyp_features)
                        key = (lm_state, tuple(coverage))
                        if key not in stacks[covered] or stacks[covered][key].logprob < logprob:
                            stacks[covered][key] = new_hypothesis
        if nbest_size > 1 :
            for top_hyp in sorted(stacks[-1].itervalues(), key=lambda h: -h.logprob)[:nbest_size]:
                assert(top_hyp.coverage.count(1) == len(top_hyp.coverage))
                temp =  str(s) +" ||| " + str(extract_english(top_hyp)) + " ||| " + " ".join([str(value) for value in top_hyp.features])
                #print temp
                result.append(temp);
        else:
            assert(len(stacks[-1]) > 0)
            # get top and backtrack
            top = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
            assert(top.coverage.count(1) == len(top.coverage))
            #print extract_english(top)
            result.append(extract_english(top))

            if verbose:
                lm_logprob=top.features[0]
                tm_logprob=sum(top.features[2:6])
                sys.stderr.write("s = %d, LM = %f, TM = %f, Total = %f\n" % (s, lm_logprob, tm_logprob,top.logprob_with_fc))
    return result

def calculate_bleu(bleu_script, trans, gold):
    with open(trans, 'r') as infile:
        mb_subprocess = Popen(['perl', bleu_script, gold], stdin=infile, stdout=PIPE)
        stdout = mb_subprocess.stdout.readline()
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        assert out_parse is not None
        bleu_score = float(out_parse.group()[6:])
        mb_subprocess.terminate()
        return bleu_score

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    #basedir is location where data files are
    #basedir = "/usr/shared/CMPT/nlp-class/project";
    basedir=".."
    optparser.add_option("-i", "--input", dest="input", default= basedir + "/dev/all.cn-en.cn", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default= basedir + "/large/phrase-table/dev-filtered/rules_cnt.final.out", help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default= basedir +"/lm/en.gigaword.3g.filtered.train_dev_test.arpa.gz", help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default= no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=5, type="int", help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=10, type="int", help="top stack size (default=1)")
    optparser.add_option("-d", "--distortion-limit", dest="d", default=6, type="int", help="top distortion limit (default=1)")
    optparser.add_option("-p", "--distortion-penalty", dest="p", default=0.9, type="float", help="Distortion penalty (default=0.9)")
    optparser.add_option("-b", "--beam-width", dest="b", default=2, type="int", help="Beam width parameter (default=10)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
    optparser.add_option("-w", "--weights", dest="weights", default="./uniform.weights", help="filename having initial weight")
    optparser.add_option("--nbv", "--nbest_value", dest="nbest_value", default=1, type= int, help="N best value")
    optparser.add_option("-g", "--gold", dest="gold", default=basedir+"/test/all.cn-en.en", help="Reference 1")
    optparser.add_option("-o", "--output", dest="output", default= "./translated.out", help="Translated English sentence.")
    optparser.add_option("--bs", "--bleu_script", dest="bleu", default= "./multi-bleu.perl", help="Multi-Bleu perl.")
    optparser.add_option("--nbest", "--nbest", dest="nbest", default="./nbest.txt", help="N-best file")
    optparser.add_option("--lw", "--learned_weight", dest="learned", default="./learned.weights", help=" Learned weight")

    opts = optparser.parse_args()[0]
    start_time = time.time()
    french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]
    
    lm = models.LM(opts.lm)    
    tm = models.TM(opts.tm, opts.k);
    ibm_model = {}
    sys.stderr.write("Loading IBM Model 1 Score..")
    
    with open('./IBMModel1_cn_en.pickle', 'rb') as handle:
        ibm_model = pickle.load(handle)
    sys.stderr.write("Completed IBM Model 1 Score..")
    
    handle_unk_words(french, tm)
    if opts.weights is None :
        weights = [1/float(feat_len) for _ in xrange(feat_len)]    #Uniform
    else:
        weights = [float(line.strip()) for line in open(opts.weights,'r')]
    # normalize weights
    
    # is this required ??
    s = sum(weights)
    weights = [w/float(s) for w in weights]
            
    if opts.nbest_value > 1:
        # if nbest generator
        if os.path.exists(opts.nbest): 
            os.remove(opts.nbest)
        weights = [1/float(feat_len) for _ in xrange(feat_len)]    #Uniform
        
        for epoch in range(3):# 3 epoch or 
            
            nbest=decode(french, tm, lm, ibm_model, opts.s, opts.d, opts.p, opts.verbose, opts.b, weights, opts.nbest_value)
            sys.stderr.write("Decoding done for epoch %d...\n" % epoch)
            
            output =  open(opts.nbest, 'w') 
            output.write("\n".join([line for line in nbest]));        
            output.close() 
            
            sys.stderr.write("Learning starting for epoch %d...\n" % epoch)
            weights = subprocess.check_output(['python', 'learn.py', '-i', opts.nbest, '--num', str(opts.num_sents)])
            #weights = subprocess.check_output(['python', 'learn.py', '-i', opts.nbest, '--num', str(opts.num_sents), '--th', opts.learned])
            # write in learned weights file
            with open(opts.learned,'w') as learned:
               learned.write(weights); 
               
            weights = [float(line.strip()) for line in open(opts.learned)]
            
             # normalize weights is this required?
            s = sum(weights)
            weights = [w/float(s) for w in weights]
            
            sys.stderr.write("Learning completed for epoch %d, and weights are %s...\n" % (epoch,weights))
    else : 
        # if decoder than
        if os.path.exists(opts.output):
            os.remove(opts.output)
        
        best = decode(french, tm, lm, ibm_model, opts.s, opts.d, opts.p, opts.verbose, opts.b, weights, opts.nbest_value)
        output =  open(opts.output, 'w') 
        output.write("\n".join([line for line in best]));        
        output.close()
               
        print calculate_bleu(opts.bleu, opts.output, opts.gold)  
         
    sys.stderr.write("\nTook %s seconds\n" % (time.time() - start_time))
