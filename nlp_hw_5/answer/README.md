
Pseudocode
------------------

Parameters:
    tau: samples generated from n-best list per input sentence (set to 5000)
    alpha: sampler acceptance cutoff (set to 0.1)
    xi: training data generated from the samples tau (set to 100)
    eta: perceptron learning rate (set to 0.1)
    epochs: number of epochs for perceptron training (set to 5)

for each sentence i:
    collect all the n-best outputs for i
    for each candidate c in the n-best list:
        compute the bleu score b (using bleu.py) for c
        append (c,b) to nbests[i]

for i = 1 to epochs:
    for nbest in nbests:
        get_sample():
            initialize sample to empty list 
            loop tau times:
                randomly choose two items from nbest list, s1 and s2:
                if fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha:
                    if s1.smoothed_bleu > s2.smoothed_bleu:
                        sample += (s1, s2)
                    else:
                        sample += (s2, s1)
                else:
                    continue
            return sample
        sort the tau samples from get_sample() using s1.smoothed_bleu - s2.smoothed_bleu
        keep the top xi (s1, s2) values from the sorted list of samples
        do a perceptron update of the parameters theta:
            if theta * s1.features <= theta * s2.features:
                mistakes += 1
                theta += eta * (s1.features - s2.features) # this is vector addition!
return theta

