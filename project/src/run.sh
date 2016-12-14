#!/bin/sh
# basedir is location where data files are
basedir=/usr/shared/CMPT/nlp-class/project
python decoder.py -k 5 -s 10 -d 6   \
				 --nbv 100 \
				 -i ${basedir}/dev/all.cn-en.cn

echo "Learning weight Done."

python decoder.py -k 5 -s 10 -d 6  \
				-w ./learned.weights \
				-i ${basedir}/test/all.cn-en.cn \
				-g ${basedir}/test/all.cn-en.en  \
				-t ${basedir}/large/phrase-table/test-filtered/rules_cnt.final.out \
				-o ./translated.out			
echo "Decoding Done"
python score-reranker -r ${basedir}/test/all.cn-en.en  < ./translated.out
echo "Scoring done."