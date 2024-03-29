---          
title: 解析詞向量 - 目錄          
categories:          
 - WordEmb101          
tags: 解析詞向量          
---          
最近這段時間，新出了不少很強大的NLP模型，比如elmo bert gpt xlnet什麼的      
這些模型刷著各種榜單，萬用又強大      
但當你跑去實作，就會悲慘地將batch size越調越小。也感慨自己3GB的1060實在是雞肋，硬件資源不足之下，可以說是與這些酷炫模型緣分不足。    
其實在資源的限制下，我們還有不少的方法也可以達到不錯的效果，比如之前很熱門的Word2Vec    

在之前的日子，做NLP的肯定繞不開詞向量    
網絡上都有著大量介紹，只可惜資料總歸零碎，總有些盲人摸象的感覺    
因此，希望在此做一個全面地整理，探尋一下詞向量的本質和應用   

這裏作爲文章的目錄，方便大家查閲和概覽這個系列的内容    
如果有不清楚或者缺失的地方，歡迎大家留言~    

才疏學淺，有錯漏請務必指正  
    
這個系列的文章主要内容：    
1 - 引子    
    
- [詞向量的發展史 - word2vec是怎麼一步步想出來的](https://voidful.github.io/voidful_blog/wordemb101/2019/07/17/word-emb-1/)     
    
2 - 細節    
    
- [解析詞向量2 word2vec訓練細節](https://voidful.github.io/voidful_blog/wordemb101/2019/07/25/word-emb-2/)   
    - word2vec訓練細節-兩套訓練方法/兩個優化方法/兩個詞向量？    
    - 選哪一個訓練和優化方式    
    
3 - 特性    
    
[解析詞向量3 word2vec有趣的特性](https://voidful.github.io/voidful_blog/wordemb101/2019/08/02/word-emb-3/)
    - 解析word2vec 的特性 - Magnitude/Cos similarity/king-women=Queen    
    - fastText - Word2Vec後續改進，處理OOV(新詞)    
    
4 - 本質    
[解析詞向量 4 word2vec的本質和Glove](https://voidful.github.io/voidful_blog/wordemb101/2019/08/28/word-emb-4)   
    - 從現象看本質 - word2vec也是PMI矩陣分解    
    - Glove - 從另外的角度思考    
    
5 - 應用    
[解析詞向量 5 網絡資源與句子向量](https://voidful.github.io/voidful_blog/wordemb101/2019/08/28/word-emb-5)  
    - 各種預訓練詞向量與code    
    - 利用詞向量建立句子向量    
    
6 - 花樣    
    
- 詞向量做翻譯    
- 詞向量找關鍵詞    
- 衡量詞相似的各種花樣    
- 萬物皆可embedding    

