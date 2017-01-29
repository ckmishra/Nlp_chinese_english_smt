# nlp_hw
Implementation of Chinese to English Statistical Machine translation System.
Project report inside project/src folder.
Also, Individual homework directory contain readme.

ABSTRACT:
This paper is for Chinese to English statisti-cal machine translation system with emphasison  feedback  exchange  between  decoder  andreranker. We have used future cost based beamsearch decoder and PRO reranking algorithmwith ordinal regression to build our decoder-reranker system. Our idea here is to use signif-icant features such as language model, trans-lation model and alignment score for our de-coder and use feedback loop from the rerankersystem  to  improve  translation  quality.Wehave improved on the PRO algorithm by com-bining ordinal regression for better learning inthe reranking phase of the system. Our experi-ment shows +3.5 BLEU score improvement ascompared to baseline system having only de-coder.  We have used Moses tokenizer whichfurther improved the BLEU score by +2.0.
