Doc
-----------------

We got better improvement with IBM model 1.
							Test
------------------------------------------|
| Baseline     			 |  24.46		  |
|+ IBM Model 1			 |	25.23		  |
|+ IBM Model 1 + Ranking |	26.55		  |

We also added other features such as word count, untranslated word and balanced double quotes. However, these feature are not helping much in improving BLUE score. 

We attached our reranking.py along with alignment features.
"align.train.feat" = IBM model 1 score on train.nbest 
"align.test.feat"  = IBM model 1 score on test.nbest

Files :
for reproducing 26.55 Blue score use below files on our /answer/rerank.py. 
learned.weight
output - best output

Other approaches and further analysis:

The problem here is not a linearly splittable problem, dividing the training examples into good and poor examples.

Successful in making the model linear:
Considered only training sentences with 300 candidate sentences for training. 
However, this does not cover the information for the sentences with less candidate sentences(shorter sentences).The Bleu score for the this
approach was 25.98.

