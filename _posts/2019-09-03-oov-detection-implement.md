---                                       
title: 實作 少量資料，高效率 一個找新詞的方法                                       
categories:                                       
 - Implement                                       
tags: 實作                                       
---                                       
                             
最近有一篇論文，探討現在中文NLP還需不需要斷詞     
《Is Word Segmentation Necessary for Deep Learning of Chinese Representations? 》     
發現在深度學習中 以字建立的模型 比 以詞建立的模型 結果要好。但詞模型的表現不好，有原因是因爲沒有解決Out of vocabulary word(OOV，新詞)的問題。加上現在還有應用是基於詞來分析的 - 比如說討論區熱點，詞雲等……     
可見，斷詞裏面新詞的問題依然存在。有沒有一個簡單有效的方法找到新詞呢？成爲本文想要探討的問題，提出一個新的方法，有以下改善：     
     
- 資源占用低     
- 不需大規模語料就能得到理想效果     
- 不需考慮閥值的設置     
     
文末也會有code，希望可以一起改善。     

### 以前新詞發現的方法和缺陷     
在以往新詞發現的方法中，通常也會提及 PMI 和 entropy 兩個指標     
     
PMI用以衡量兩個字 是在一起還是剛好碰到一起，也就是兩個字出現，剛好碰到一起的機率$$$$$$P(A) \times P(B)$$ 和 詞的機率$$P(AB)$$ 的比值     
PMI : $$log\frac{P(AB)}{P(a) \times P(b)}$$     
     
而Entropy則是基於一個詞應該可以用在不同的場景，因此看這個詞的左右搭配是否豐富，越豐富的搭配越可能是詞。而衡量左右的詞是否豐富，Entropy是一個很切合的指標。     
Entropy : $$H(X) = -\sum p(X)\log p(X)$$     
     
這兩個方法都有缺點：     
Entropy要統計左右詞頻，導致時間複雜度大大提高     
PMI和Entropy都需要設置一個閾值加以篩選，而這個閾值由於語料大小，語境各種因數而變化，不容易調好     
文本量越大效果越好，因此需要大量的文本作爲基礎，然而文本越多，效率越差     
     
這些問題都很不好解決，因此，可以換一個全新思路來看看這個問題     

### 重新出發，尋找另外的方法   
我們從本源出發，詞的定義是什麽？     
     
根據 中央研究院  CNS14366中文分詞原則     
句子與文本的理解建立在詞義的組合上     
具有獨立意義，且扮演固定詞類的字串視為一分詞單位     
也就是說，詞語詞之間是獨立的，以字為單位看，     
詞語之間 跟 詞本身 有著統計上的差異     
詞語之間的關聯度 相對 詞本身的關聯度 會低     
兩者相互對比可以找出詞的邊界，也是句子獨立部分     
     
我們用一張圖直觀解釋一下：     
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img1)     
     
     
句子中詞與詞之間的邊界，詞頻會大幅下降，這一個drop down是一個很好判斷是否為詞的訊號。     
單純用drop down的大小對比會陷入難以設定閾值，資料改動容易導致閾值大幅度改變等問題，不夠穩定可行     
考慮到句子產生可以說是不斷在文末加入字的過程，頻數的變化，可改用 前文 和 前文+下一字 的比值來表達，而這個比值，也剛好是產生下一個字的條件概率。     
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img2)         
     
     
在條件概率下，值越小表示越可能是邊界位置，也就越可能是詞。     
從上圖可見也發現，條件概率等於1基本上不可能是詞，因此也可以過濾掉。     
     
也就是說，我們搭建一個基於 {Ngram:frequency}的dictionary，就可以做到新詞發現這個事情，第一步的預處理可以篩選不可能是詞的結果，進一步增加效率：     
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img3)     
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img4)        
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img5)        
     
     
在預處理之後，對著剩下的詞表算一下條件概率，做一些過濾(按照實際的需求來做，最通用可以過濾條件概率為1的結果)，剩下的就是我們想要我們的詞。     
一段779字的文本作爲輸入，我們看看結果如何：     
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img6)       
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img7)        
     
     
可以看出，僅僅輸入1000字左右的文本就足以得到不錯的結果。     
當然啦，得出來的結果不會全是好的結果，我們可以用更多的規則過濾，也可以根據應用有相對應的取捨。     
     
我們也可以借此做出熱點分析的應用，抓取討論區一段時間内的文章，由於這個方法文本量不需要很多，因此這一段短時間還可以設置比較短。找出這一段時間内的新詞，就可以看到這段時間的熱點會有什麽！     
     
以Dcard軟體工程師版為例子，可以得到這樣的結果：     
     
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_oovdet/img8)        
     
     
以下是Colab demo code 和 GitHub source code     
     
[Colab Demo + Dcard Hot Topic Analysis](https://colab.research.google.com/drive/1n-JVX7XPupWz3RuoOOMv-1sQAhMo3sQo)     
     
[Github](https://github.com/voidful/Phraseg)     
     