---                                  
title: 解析詞向量 5 網絡資源與句子向量                                  
categories:                                  
 - WordEmb101                                  
tags: 解析詞向量                                  
---                                  
                        
這篇文章講的是：                
- 網絡上預訓練的詞向量資源       
- 簡單取得句子向量的方法       


我們要使用word2vec，從零開始爬資料，預處理，訓練十分繁瑣。   
因此推薦一開始先試一些預訓練好的詞向量：   
   
[https://github.com/liuhuanyong/ChineseEmbedding](https://github.com/liuhuanyong/ChineseEmbedding)   
   
[https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)   
   
[https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Kyubyong/wordvectors)   
   
[https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/candlewill/Chinsese_word_vectors)   
   
   
我們有詞向量后，用以分析句子和文本，通常會拿句子的向量做文本分析。   
很直觀的做法，是用一個神經網絡訓練一個語言模型，其實也有不少簡單直接的方法可以得到和語言模型差不多的結果。拿到詞向量不需要訓練就可以直接用，考慮到方便和效率，很建議一開始先試這些方法的效果。   
   
來自於   
《Baseline Needs More Love:On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms》   
有三個方法：   
   
1. SWEM-aver 每一維取平均   
2. SWEM-max 每一維區最大值   
3. SWEM-concat 將1，2兩個方法concat起來   
4. SWEM-hier 用窗口捕捉上下文訊息，將windows size内的詞向量做average pooling，拿到每一塊Windows size的結果之後，做max pooling拿到最終的結果   
   
以上方法的code和使用方法，放在在nlp2 project裏面，有興趣可以瞭解一下：   
   
[https://github.com/voidful/nlp2#vectorize]   
   
   
Reference：   
Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms https://arxiv.org/pdf/1805.09843.pdf   