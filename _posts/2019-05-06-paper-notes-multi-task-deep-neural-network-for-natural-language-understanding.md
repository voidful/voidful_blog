---
title: 語料不夠怎麼辦，Bert在多任務上預訓練說不定有用
categories:
 - Paper Reading
tags: 論文解析
---

**Multi-Task Deep Neural Networks for Natural Language Understanding**

這篇paper主要是在bert之上再做的改進

Bert先用Masked LM和Next Sentence Prediction兩個方式預訓練representation

![Masked LM](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img1)

![Next Sentence Prediction](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img1)


再用這個representation用finetune的形式分別訓練各種task

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img3)


這個時候，就有了一個想法，這些下游的task沒有必要分開訓練

就好比學會判斷句子是否有反諷這個過程，其實也了解到某些閱讀理解的方法，反之，學會閱讀理解之後做反諷判斷也會更加得心應手

將多個task聯合去學習，從而得到更加健壯的模型，將這樣的想法結合到原始的bert，也就是這篇論文所要做的


![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img4)


從上圖可見，架構沒有很複雜，
首先通過bert 的input 將文本encode
得到一個embedding
其所對應的code是bertModel這一部分
之後通過multitask調整這個embedding
這樣finetune完之後，可以拿這個embedding去訓練新的task

麻煩的地方在於怎麼做multitask，將他們聯合起來訓練
通常multitask learning 會采取以下兩種做法：
Hard parameter sharing / Soft parameter sharing

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img5)

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img6)


Hard parameter sharing 如圖所示，將每個task之間共用隱藏層，但輸出層則會分別處理，這可以確保不同的輸入根據task的不同有不同輸出。
Soft parameter sharing則是每個模型都有獨立參數，通過正則化的手段讓不同模型的參數拉到同一個空間内。這可以讓模型之間比較接近之餘在不同任務之間也可以有所不同。

按照Bert的情況，選用Hard parameter sharing的方式是比較合適的，bert已經證明它fine-tune之下的强大之處，我們在fine-tune階段做更多的task，説不定讓這個embedding能力更強。

直觀地，可以先訓練task a再訓練task b
但這種方式，越往後訓練的task由於當前bp的調整而有更好的表現，但之前的task卻因此而影響表現，整體而言很難達到最佳

因此這篇paper的想法是將這些task拆細一些，拆成一個個batch，再打亂其順序

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img7)


每次抽取一個batch，根據其屬於的任務loss

用這個方法在不同的task上fine-tune，看看其效果如何。
這裏用GLUE這個dataset，其包含9個NLU的task，可以多方面衡量這個模型的能力
可見其在每一個task都有提升

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img7)


雖然提升是有提升，但其實只有1~2%，最高也只有5%左右，感覺這個方法用處不大啊
但其實，其强大的地方是在於，在面對一個新的task，在缺乏足夠語料的情況下，卻可以得到極爲不錯的效果。
做完GLUE task之後，引入以下兩個新task，分別取0.1%，1%，10%和100%的資料，模擬資料不足的情況下，模型的能力

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_mtdnnfnlu/img8)


效果可以說十分驚艷，在少量的data上可以得到極高的結果，證明這比一般的bert有更加强大的一般化能力。

後續來看，Microsoft還將這個方法用到CoQA上，得到很不錯的結果

https://www.microsoft.com/en-us/research/blog/machine-reading-systems-are-becoming-more-conversational/

![Overview of the multistage multitask fine-tuning model](https://www.microsoft.com/en-us/research/uploads/prod/2019/05/coqa_figure_2-1024x704.jpg)






