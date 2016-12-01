import optparse, sys, os
from collections import namedtuple
import math
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import re
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def matched_para(sentence):
    #for n, line in enumerate(open(nbest)):
    #    (i, sentence, features) = line.strip().split("|||")   
        count = 0
        for i in sentence:
            if i == "(":
                count += 1
            elif i == ")":
                count -= 1
        return count 
    
def quotationMatch(sentence):
        count = 0
        for i in sentence:
            if i == '\"':
                count += 1
            elif i == '\"':
                count -= 1
        return count 
        
if __name__ == '__main__':
    
    optparser = optparse.OptionParser()
    optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("../data", "train.nbest"), help="N-best file")
    optparser.add_option("-f", "--train_fr", dest="train_fr", default=os.path.join("../data", "train.fr"), help="French Training data")

    (opts, _) = optparser.parse_args()
    
    src = [line.strip().split() for line in open(opts.train_fr)]
    
    for n, line in enumerate(open(opts.nbest)):
            (i, sentence, features) = line.strip().split("|||")    
            translated_len_feat = len(sentence.strip().split())
            untranslated_feat = [word for word in sentence.strip().split() if not(is_ascii(word))] 
            #yield str(math.log10(translated_len_feat))
            #yield  str(translated_len_feat) +" "+str((math.log10(translated_len_feat)-math.log10(len(src))))+" "+str((translated_len_feat-len(src))) +" "+str(len(untranslated_feat))
            #print str(math.log10(translated_len_feat) - math.log10(len(src[int(i)]))) +" "+ str(math.log10(len(untranslated_feat)+1))
            print   str(quotationMatch(sentence)) + " " + str(matched_para(sentence))

    #str((translated_len_feat)) +" "+ str((len(untranslated_feat))) +" "

       
        
