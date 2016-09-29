
Your documentation
------------------
Steps :
1. We started with baseline approach given in assignment.
2. After that, we implemented Bigram model however our score haven't improved as compared to Unigram model.
3. We applied smoothing on Unigram model. Tried various smoothing approach such as Add one.
4. In given "input" data, we observed there are lot of numbers and we handled by giving higher probability.
5. Step 4 has improved our score, further we applied Mercer Smoothing on Bigram model.
6. For Mercer smoohing, we ran various run by changing "lambda" starting from 0.1 to 0.9, finally choosen lambda giving best result.   