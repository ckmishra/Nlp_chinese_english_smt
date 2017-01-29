
#!/bin/sh

#basedir=/usr/shared/CMPT/nlp-class/project
basedir=".."
# utility function 
# need to download tokenizer from https://github.com/moses-smt/mosesdecoder and specify below
moses_tokenizer=/Users/apple/nmt_project/mosesdecoder/scripts/tokenizer/tokenizer.perl
tokenize () {
    inp=$1
    out=$2
    lang=$3
    if [ -e "${out}" ]; then
        echo "[${out}] exists"
    else 
        echo "...tokenizing $inp with $lang tokenizer"
        perl ${moses_tokenizer} -l ${lang} -threads 8 < ${inp} > ${out} 
    fi 
}


# tokenize english
tokenize ${basedir}/test/all.cn-en.en0 ${basedir}/test/all.cn-en.en.tok.en0 en 
tokenize ${basedir}/test/all.cn-en.en1 ${basedir}/test/all.cn-en.en.tok.en1 en 
tokenize ${basedir}/test/all.cn-en.en2 ${basedir}/test/all.cn-en.en.tok.en2 en 
tokenize ${basedir}/test/all.cn-en.en3 ${basedir}/test/all.cn-en.en.tok.en3 en 

# tokenize output
tokenize ${basedir}/src/translated.out_9.7 ${basedir}/src/translated.out_9.7.tok en
 
