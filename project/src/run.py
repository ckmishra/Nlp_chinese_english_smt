import bleu
import optparse,sys, time
from nltk.translate import bleu_score
import subprocess
from subprocess import  PIPE, Popen
import re



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
    basedir = "..";
    optparser.add_option("-i", "--trans", dest="trans", default=  "./translated.out", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("--r1", "--ref1", dest="ref1", default= basedir + "/test/all.cn-en.en0", help="Reference 1")
    optparser.add_option("--r2", "--ref2", dest="ref2", default= basedir + "/test/all.cn-en.en1", help="Reference 2")
    optparser.add_option("--r3", "--ref3", dest="ref3", default= basedir + "/test/all.cn-en.en2", help="Reference 3")
    optparser.add_option("--r4", "--ref4", dest="ref4", default= basedir + "/test/all.cn-en.en3", help="Reference 4")
    
    optparser.add_option("-g", "--gold", dest="gold", default=basedir+"/test/all.cn-en.en", help="Reference 1")
    optparser.add_option("-b", "--bleu_script", dest="bleu", default= "./multi-bleu.perl", help="Multi-Bleu perl.")
  
    opts = optparser.parse_args()[0]
    
    start_time = time.time()
    
    # testing bleu score
    ref1 = [line.strip().split() for line in open(opts.ref1)]
    ref2 = [line.strip().split() for line in open(opts.ref2)]
    ref3 = [line.strip().split() for line in open(opts.ref3)]
    ref4 = [line.strip().split() for line in open(opts.ref4)]
    
    trans = [line.strip().split() for line in open(opts.trans)]
    final = 0.0
    local = 0.0
    #local+= bleu.getBleuScoreForMultiRef(trans[n], [ref1[n], ref2[n], ref3[n], ref4[n]]);
    for n in range(len(trans)):

      final += bleu_score.bleu([ref1[n], ref2[n], ref3[n], ref4[n]], trans[n], [0.25, 0.25, 0.25, 0.25])
    
    print "Using NLTK : " , final
    print "Using Given : "
    print calculate_bleu(opts.bleu, opts.trans, opts.gold)
    