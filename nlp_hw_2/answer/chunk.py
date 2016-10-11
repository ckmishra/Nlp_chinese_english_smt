"""

You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.

Each element of train_data is a (labeled_list, feat_list) pair. 

Inside the perceptron training loop:

    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list

    - If the output is incorrect then we have to update feat_vec (the weight vector)

    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)

    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id

    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)

    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case

    - feat_vec[feature_id] is the weight associated with feature_id

    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product

    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value

    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.

"""
from collections import Counter   
import sys, optparse, os
from datetime import datetime
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import perc
from collections import defaultdict

def perc_train(train_data, tagset, numepochs):
    starttime = datetime.now()
    feat_vec = defaultdict(int)
    default_tag = tagset[0]
    sigma = defaultdict(int)
    gamma = defaultdict(int)
    # insert your code here
    # please limit the number of iterations of training to n iterations
    for i in range(numepochs):
        epochstarttime = datetime.now()
        numOfError = 0;
        argMaxoutput = [];
        for (labeled_list, feat_list) in train_data:
            argMaxoutput = perc.perc_test(feat_vec, labeled_list, feat_list, tagset, default_tag)  
            expectedOutput = [element.split(" ")[2] for element in labeled_list]
            for j in range(len(expectedOutput)):
                trueLabel = expectedOutput[j]
                argMaxLabel = argMaxoutput[j]
                trueLabel_prev = expectedOutput[j-1];
                argMaxLabel_prev = argMaxoutput[j-1];
                if (trueLabel != argMaxLabel) :
                    numOfError = numOfError + 1;
                    for feat in feat_list[j*20:j*20+20] :
                        if (feat =="B") & (j > 0):
                            trueLabel_prev = expectedOutput[j-1];
                            argMaxLabel_prev = argMaxoutput[j-1];                  
                            feat_vec["B:"+trueLabel_prev,trueLabel] = feat_vec["B:"+trueLabel_prev,trueLabel] + 1
                            feat_vec["B:"+argMaxLabel_prev,argMaxLabel] = feat_vec["B:"+argMaxLabel_prev,argMaxLabel] - 1
                            #sigma["B:"+trueLabel_prev,trueLabel] = sigma["B:"+ trueLabel_prev,trueLabel] + feat_vec["B:"+ trueLabel_prev,trueLabel]
                            #sigma["B:"+argMaxLabel_prev,argMaxLabel] = sigma["B:"+argMaxLabel_prev,argMaxLabel] + feat_vec["B:"+argMaxLabel_prev,argMaxLabel]
                        else :
                            feat_vec[feat,trueLabel] =  feat_vec[feat,trueLabel] + 1;
                            feat_vec[feat,argMaxLabel] = feat_vec[feat,argMaxLabel] - 1;
                            #sigma[feat,trueLabel] =  sigma[feat,trueLabel] + feat_vec[feat,trueLabel];
                            #sigma[feat,argMaxLabel] = sigma[feat,argMaxLabel] + feat_vec[feat,argMaxLabel];
                
                elif (j > 0) & (trueLabel == argMaxLabel) & (trueLabel_prev != argMaxLabel_prev):
                        feat_vec["B:"+trueLabel_prev,trueLabel] = feat_vec["B:"+ trueLabel_prev,trueLabel] + 1
                        feat_vec["B:"+argMaxLabel_prev,argMaxLabel] = feat_vec["B:"+argMaxLabel_prev,argMaxLabel] - 1
                        #sigma["B:"+trueLabel_prev,trueLabel] = sigma["B:"+ trueLabel_prev,trueLabel] + feat_vec["B:"+ trueLabel_prev,trueLabel]
                        #sigma["B:"+argMaxLabel_prev,argMaxLabel] = sigma["B:"+argMaxLabel_prev,argMaxLabel] + feat_vec["B:"+argMaxLabel_prev,argMaxLabel]
                '''
                elif (j > 1) & (trueLabel == argMaxLabel) & (trueLabel_prev == argMaxLabel_prev) & (expectedOutput[j-2] == argMaxoutput[j-2]):
                        feat_vec["B:"+expectedOutput[j-2],trueLabel_prev,trueLabel] = feat_vec["B:"+ expectedOutput[j-2],trueLabel_prev,trueLabel] + 1
                        feat_vec["B:"+argMaxLabel_prev,argMaxLabel] = feat_vec["B:"+argMaxoutput[j-2],argMaxLabel_prev,argMaxLabel] - 1
                        #sigma["B:"+trueLabel_prev,trueLabel] = sigma["B:"+ trueLabel_prev,trueLabel] + feat_vec["B:"+ trueLabel_prev,trueLabel]
                        #sigma["B:"+argMaxLabel_prev,argMaxLabel] = sigma["B:"+argMaxLabel_prev,argMaxLabel] + feat_vec["B:"+argMaxLabel_prev,argMaxLabel]
                '''
            #sigma = dict(Counter(sigma)+Counter(feat_vec))
            
            for key in feat_vec:                             
                sigma[key] = sigma[key]  + feat_vec[key];
            
        epochendtime = datetime.now()
        print "Number of error in Epoch", i+1," ", numOfError," Time Taken:", epochendtime-epochstarttime  
    
    for key,value in sigma.items():
        gamma[key] = value/(numepochs*len(train_data))
        #gamma[key] = value/(numepochs)

    endtime = datetime.now()
    print "Total Time taken to train:", endtime-starttime
    return gamma

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("../data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("../data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("../data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(15), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("../data", "test.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    tagset = perc.read_tagset(opts.tagsetfile)
    print >>sys.stderr, "reading data ..."
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    print >>sys.stderr, "done."
    feat_vec = perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)

