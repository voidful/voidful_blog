---
title: Multi-Task的最高境界是沒有Multi-Task? 解析OpenAI GPT2背後的想法
categories:
 - Paper Reading
tags: 論文解析
---

來自OpenAI GPT-2 的 Language Models are Unsupervised Multitask Learners   
來自OpenAI的 Generative Modeling with Sparse Transformers   
   
上一文中介紹Bert Multi-Task上的進展（有興趣可以去看看 [link](https://voidful.github.io/voidful_blog/paper%20reading/2019/05/06/paper-notes-multi-task-deep-neural-network-for-natural-language-understanding/) )   
此後，進一步看multi-task在NLP上面的各種case，發現OpenAI的GPT和其後續有著十分驚艷的想法   

在Deep Learning上的multi-task主要有以下兩種做法，soft parameter sharing在layer之間做normalization，而hard parameter sharing則是底層layer公用，根據不同的task設計不同的layer   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_togast/img1)   
   
   
參考最近的Bert和GPT，其實背後就是用了hard paramenter sharing的方法。   
hard parameter sharing在實作上有好幾種做法   
   
一種是上文介紹，微軟MT-DNN的做法，將不同的task的資料切成batch，再將這些batch混合打亂，這樣每次batch更新就只會是一個task的loss。   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_togast/img2)   
   
   
一種常見的做法，就是一個batch裏可能有著好幾個task的資料，每一筆資料根據其task算出loss，而這個batch的loss則是將這些loss加起來。Bert和GPT1在pre-training的時候就是采用這個方法的。   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_togast/img3)   
   
   
**減法來了**   
這兩種做法都需要根據不同的task設計一個output layer，task越多，就會一直增長下去，最終會十分臃腫   
因此再有一個想法，是將Task的訊息也放到模型中，給一個task signal説明目前是在什麽task，然後讓其輸出task的結果，放到Bert來看會是這樣   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_togast/img4)   
   
   
這種做法在實在上更加容易，但顯然有人並不滿足於此，我們甚至不需要給task signal，其實也可以做到   
   
nlp的任務中，supervised learning 是根據我們的objective function使得模型達到global minima，通俗地說，就是讓model輸出得到我們期望的結果，比如在閱讀理解中，我們輸入文章和問題，模型會知道我們預測的結果，而模型學習的過程就是根據我們提供的輸入輸出得到他們的關聯性。   
   
但如果我們沒有告訴模型輸入和輸出，模型其實也能夠學到其中關聯性吧！   
爲什麽這樣想呢，如果模型在閱讀理解中，在我們輸入文章和問題中，模型都曾經看過類似的訊息，也就會就知道應該怎麽輸出。   
   
爲了讓這些事情成功，模型需要看過大量的文本，有著各種不同的用法和表達方式的文本，以此學到不同的任務訊息。訓練一個超大文本下的，unsupervised 的pretraining，裏面包含著不同task的文本，比如有翻譯對照的文本，有閱讀理解及其問題答案的文本。當我們要做新的task的時候，其實不需要finetune，模型也從大量文本的pre-training中推斷出相關的答案。   
   
這個也就是OpenAI GPT-2的想法，通過大量的數據使得我們不需要提供multi-task的訊息，模型也可以學到multi-task。   
對於其驚艷的效果，也可以在這個notebook上嘗試看看   
   
https://colab.research.google.com/github/ilopezfr/gpt-2/blob/master/gpt-2-playground_.ipynb   
   
   
One more thing…   
OpenAI的想法其實不止於此，GPT-2的想法是通過 大量的數據 和 能充分捕捉/存儲訊息的網絡架構 得到强大的特徵表達能力   
OpenAI也完成對文本的建模，下一步怎麽能放過圖像和音頻呢！   
因此他們也提出了Sparse Transformers，將圖像，音頻也作爲pre-training的數據。   
輸入跟輸出的任務跟GPT，或者說language model的想法一樣，就是根據前面是sequence來預測下一個狀態會是什麽。   
爲了提高效率，適配到影片和音頻輸入，修改了Transformer的架構使得其更有效率。   
   
當這三種數據都合在一個模型中的時候，語音轉文字，按圖講故事，文本轉語音各種task説不定有新的突破，離通用的AI將更進一步。   





