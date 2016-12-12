import optparse, sys, os
from collections import namedtuple
import math
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

def is_ascii(s):
    return all(ord(c) < 128 for c in s)
 
def quotationMatch(sentence):
        count = 0
        for i in sentence:
            if i == '\"':
                count += 1
        return (count % 2)

def findFullstop(sentence):
    try:
        if sentence.index(".") < len(sentence)-1:
            return -1
    except:
        return 0
        exit()
    return 1    

