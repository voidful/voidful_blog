---      
title: Bert 怎麽用？bert各種使用上的疑問和細節      
categories:      
 - Implement      
tags: 實作      
---      

Bert出來好一段時間，使用過程中或多或少會有一些疑問：   
   
- 如果不做finetune而是傳統的方法會怎麼樣？   
- 只拿最後一層真的是最好的選擇嗎？   
- bert在中文上怎麼樣可以做到更好？   
- 超過512個字應該怎麼樣處理？   
- bert可以做文本生成嗎？   
- Bert做多任務?   
- Bert可以用在什麽Task上面呢？   
- MaskLM和NextSentencePrediction兩種訓練方式應該怎麼關聯到我們的任務上？   
   
在此希望對這些問題探討看看~
   

## 回顧Bert的訓練過程   
Bert 是Multi-task的方式訓練，其中有兩個task，一個是NextSentencePrediction - 預測兩個句子是否前後文關係；一個是MaskLM，預測被MASK起來的字   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img1)   
   
   
輸入的數據經過三層embedding后，輸入到transformer的Encoder，然後輸出兩個task的結果   
魔鬼在於細節，值得留意的是：   
在那麽多層的transformer，Bert只取最後一層作爲輸出。   
其中 MaskLM是用整個最後一層 作爲預測的輸入   
而 NextSentencePrediction卻是只用第一個輸入，也就是[CLS]作爲預測的輸入   
也就是說，[CLS]會學到句子之間的關係，會隱含句子層面的訊息   
   
兩個Task聯合更新，也是一貫mullti-task的做法，將兩個task的loss相加   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img2)   
   
   
## MaskLM和NextSentencePrediction兩種訓練方式應該怎麼關聯到我們的任務上？   
在看過Bert的訓練過程，可以發現，對於句子級別的任務更加適合用[CLS]來處理   
字級別的任務，則可**MaskLM輸出接FeedForward**   
其中一個例子是 Fine-tune BERT for Extractive Summarization 所做的抽取式摘要   
就是將句子用[CLS]跟[SEP]相隔作爲輸入，然後取[CLS]丟到summarization layer得到預測   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img3)   
   
   
也是考慮到[CLS]可以學到句子層面的關係，而采用這樣的架構   
   
## 只拿最後一層真的是最好的選擇嗎？   
爲什麽Bert只拿最後一層呢?   
照理說，不同層的transformer應該會學到不同類型的語義信息，整合所有層的輸出應該效果會最好   
我們並不確定哪一層有用，哪一層沒有用，因此可以讓網絡自己去學，應該取那些訊息來用   
Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection   
就是嘗試修改Bert的架構達成這件事情   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img4)   
   
   
最後做出的結果，也是符合預期   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img5)   
   
   
由此可見，設計一個考慮各層的方法有助於提高性能。猜測由於訓練的時候是取最後一層，使得最後一層學到更多判斷語義的訊息，因此只拿最後一層，也夠用了。   
   
   
## 如果不做finetune而是傳統的方法會怎麼樣？   
傳統的方法如word2vec，通常會將詞向量作爲模型輸入的一部分，再去訓練整個模型，可以稱爲feature extraction，相對bert則提出直接finetune這個做法。   
feature extraction模型上做finetune會怎麽樣？Bert用feature extraction的方法又會怎麽樣？   
To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks   
這篇論文在ELMo和Bert就這個問題，在各種task上面比較   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img6)   
   
   
Bert 的 finetune都比feature extraction要好，而ELMo則是相反。   
按照論文的解釋，finetune會在任務跟pretrain很相近時有更加好的結果，feature extraction則是在task任務跟pretrain關係比較疏遠時，會有更加好的結果。   
據此猜測，Bert的finetune效果更好是因爲MaskLM跟NextSentencePrediction在各種NLP Task有很高的普適性，其實處理的問題也是差不多，因此Bert上finetune效果會是最好的。   
這也與我在QAnet上用bert作爲輸入embedding 對比 Bert Finetune 的實驗結果一致，Bert的fine-tune收斂更快，結果也比較好。   
   
## bert在中文上怎麼樣可以做到更好？   
Bert用在中文上，可以加入詞的訊息，讓模型達到更好的效果。   
參考baidu所做的ERNIE，其實方法簡單，就是在MaskLM的時候用詞爲單位做mask   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img7)   
   
   
當預測目標由 爾 變成 哈爾濱 後，attention得到的訊息更有層次，可以知道哈爾濱的跟其他字的聯係可以被區分開來計算。   
最終可以帶來全面的提升:   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img8)   
   
   
英文也有類似phrase的存在，其實也面臨一樣的問題。   
但詞的訊息需要預先斷詞去獲得，斷詞的各種問題也將是一個麻煩 OWO   
為了避免麻煩，Google提出另外的思路 - BERT Mask N-Gram ：用N-Gram來代替斷詞   
改善中文Bert的結果，可以試試看用WordMaskLM或NGramMaskLM的方式去finetune目前的pretraining   
   
## 超過512個字應該怎麼樣處理？   
github 上有個Issue在討論這個問題   
https://github.com/google-research/bert/issues/66   
在QA的Task上，當輸入的段落大於threshold，可以用一個sliding Windows分成幾段   
假設以下是輸入，sliding Windows設 6   
   
    the man went to the store and bought a gallon of milk tea.   
   
就會得到   
   
    the man went to the store   
    and bought a gallon of milk.   
   
然後到QA的task，若答案在第二句，則會將第一句標記為沒有答案（SQuAD 2）   
第二句則重新計算答案位置標注即可。這樣一刀切的方法未免太過粗暴了，假設答案的推斷是需要上下文訊息，這樣不就缺失了嗎   
所有又有了改善方法，切完之後，從切斷文本的一半作爲新的起點。   
用例子會比較清晰，以上面的句子，同樣設sliding Windows為 6，這個方法會得到：   
   
    the man went to the store   
    to the store and bought a   
    and bought a gallon of milk   
    gallon of milk tea   
   
有了更多的sample讓模型更能抓到前後文訊息。   
CODE:   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img9)   
   
如果不做finetune而是傳統的方法會怎麼樣？   
但對於分類的任務來説，確定data所屬的類別應該是要看過全文，切片段就判斷是武斷的，用類似QA的方法不太可行了。   
爲了得到全文的訊息，可以嘗試將向量concat起來，然後丟去輸出，也可以concat之後取平均合并，使得維度不變。   
而從我實驗的資料集來看，兩個方法都差別不大，合并取平均有某些tasks上面結果略好一些   
   
## bert可以做文本生成嗎？   
我們可以利用MaskLM的特性去生成文本，將要生成的文本都mask起來   
用teacher forcing的方法，一個一個生成。   
也可以一次過將所有Mask都預測出來。   
具體的做法和範例可以參考：   
   
[BertGenerate Github](https://github.com/voidful/BertGenerate)   
   
   
更進一步的，Bert只用了transformer的encoder，在文本生成來說顯然要結合decoder才能更好發揮！   
MASS: Masked Sequence to Sequence Pre-training for Language Generation   
就希望把decoder也拉進來，作為bert的pretrain   
訓練方法其實跟MaskLM很像，mask連續片段作為encoder的輸入，decoder負責逐個輸出預測。   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_bi1/img10)   
   
   
這個方法很像是Bert跟GPT的結合。這樣的架構也適合用在各種序列有關的任務上，如微軟最近的一篇Almost Unsupervised Text to Speech and Automatic Speech Recognition    
就是利用MASS架構，用類似Cycle gan方法train tts 同 stt   
   
## Bert可以怎麽做 Multi-Task ?   
在之前的兩篇文章中有解析相關的做法，有興趣可以移步看看   
   
[語料不夠怎麼辦，Bert在多任務上預訓練說不定有用](https://voidful.github.io/voidful_blog/paper%20reading/2019/05/06/paper-notes-multi-task-deep-neural-network-for-natural-language-understanding/)   
   
[Multi-Task的最高境界是沒有Multi-Task? 解析OpenAI GPT2背後的想法](https://voidful.github.io/voidful_blog/paper%20reading/2019/05/13/paper-notes-thinking-openai-gpt2-and-sparse-transformers/)   
   
   
## Bert可以用在什麽Task上面呢？   
根據萬物皆可awesome的定律，肯定有人收集Bert各種相關的任務   
果不其然，Github上面就有相關的awesome list   
   
[https://github.com/Jiakui/awesome-bert](https://github.com/Jiakui/awesome-bert)   
   
[https://github.com/cedrickchee/awesome-bert-nlp](https://github.com/cedrickchee/awesome-bert-nlp)   
   
   
參考自：   
Multi-Head Multi-Layer Attention to Deep Language Representations for Grammatical Error Detection   
To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks   
MASS: Masked Sequence to Sequence Pre-training for Language Generation   
Bert時代的創新：Bert應用模式比較及其它   