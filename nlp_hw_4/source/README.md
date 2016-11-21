
Results
------------------

Run using distortion limit >= 10 and stack size >= 100  and k >= 10 for better result. 
With future cost implementation, if d < 10 and s = 100 then many hypotheses are dropped earlier and having uncovered words. 
To handle above issue, allowing distortion for larger sentences (word count > 20). However, giving distortion penalty.

Our Results :
|---|-------|-------|----------|------------------- |---------------------------------------------------------------|
s	| 	k	| 	d 	|	score  |	time (in sec.)	|						comment						
--------------------------------------------------------------------------------------------------------------------|
100		20		10		-1256.62		207 
100		20		15		-1233.60		272			
100		20		20		-1229.30		296 
1000	20		10		-1238.84		1662
1000	20		15		-1223.57		2000
100		20		6		-1240.89		287			 distortion limit relaxation for larger sentences(word count>20)
100 	10 		10		-1237.40		240			 distortion limit relaxation for larger sentences(word count>20)
100		20		10		-1228.46		292			 distortion limit relaxation for larger sentences(word count>20)
100		20		15		-1228.46		290			 distortion limit relaxation for larger sentences(word count>20)
1000	20		15		-1215.83		2200	     distortion limit relaxation for larger sentences(word count>20)

Best Result = 1215.83 with stack size -s 1000 -k 20 -d 15
Best result (considering speed) = -1228.46 will be -s 100 -k 20 -d 10

Troubleshooting :
If getting below error on restricting stack size (10) and distortion limit(d=4), please comment out line#27 in get_all_phrases()
Result is quite good even though with less stack size.
Error trace : 
File "decoder.py", line 150, in <module>
    decode(french, tm, lm, opts.s, opts.d, opts.p, opts.verbose,opts.b)
  File "decoder.py", line 118, in decode
    assert(len(stacks[-1]) > 0)
AssertionError