---                                           
title: 用ALBERT測夢，順便談談文本生成的decoding方式   
categories:    
 - Implement        
tags: 實作 3line   
---      

事因小弟找不到實習，看來最近工作難找，該發展一下副業賺點錢了   
恰好在GITHUB裡面看到一個做周公解夢的Repo，輸入夢境，然後用文本生成的方式輸出其中的預兆。   
說起來，AI + 玄學 = 負負得正！！！！這件事情十分合理   
我就做一個albert解夢，然後在夜市找個位置，擺個攤位，價格低廉，說不定就年收百萬了(誤)   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/1.png)   

廢話不多說，馬上來看看怎麼把這個東西做出來！   
首先，資料集是閒逛github的時候發現的：[source](https://github.com/saiwaiyanyu/tensorflow-bert-seq2seq-dream-decoder)   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/2.png)   
資料集的輸入說夢到的內容。如：夢到發大財什麼的   
然後輸出會是這個夢境其中所蘊含的預兆。   
說真的，我蠻好奇資料集本身是怎麼來的，資料集的準確度如何。但想來這種東西根本沒有準確度可言，大致瀏覽一下，句子通順，看起來很唬爛好像就很不錯了！   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/3.png)   
好，也就是說，我們有了一份高品質的資料集！下一步就是模型的部分!   
LSTM+Seq2Seq是一套很成熟的做法，但不夠新潮，也沒有大規模的預訓練資料能讓模型有額外的知識，這些額外的知識說不定能帶給我們意想不到的結果，也會讓生成的文本更加自然通順。   
   
而最近預訓練模型，最有名的就是BERT了～   
要用BERT來做文本生成，我們可以在預訓練masklm上修改，使得BERT擁有文本生成的能力。   
masklm預測mask字時候，會根據左右文本的訊息，這就像是一個雙向的語言模型，可以根據左右的字推測中間mask起來的內容。而我們在文本生成上則是單向的，因此我們可以把BERT改成單向語言模型的形式來做到文本生成。   
然而BERT-base模型其實也蠻大的，因此我決定採用ALBERT的small來訓練，主要是為了放出來demo的時候，不要吃光機器的資源XD   
   
我們將資料集切成訓練，測試和驗證，馬上來訓練看看。   
幾個小時後結果出來了，我們跑驗證集看看效果：   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/4.png)   
   
面對不同輸入，輸出都太像了，都在說什麼運勢不錯之類的，完全沒有那種神棍的感覺   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/5.png)   
   
這個是由於，文本生成模型的目標是根據前文預測下一個字，常見的字很常被預測到，導致常見字出現機率過高。這個問題在各類型的文本生成模型上都有，最常見就是生成式的聊天機器人會經常用一些通用的回覆。   
   
   
我們這次從decoding方式入手，去解決這個問題！   
這個方法在之前文章中也有詳細介紹過，有興趣可以參考看看：   
https://blog.voidful.tech/paper%20reading/2019/05/19/paper-notes-the-curious-case-of-neural-text-degeneration/   
因此在這裡會簡單圖解，務求用簡單清晰的方式說明解決方式～   
   
首先，我們生成文本，其實就是 有前文，預測下一個字 的過程。每一次預測，我們會在我們整個詞表中，每一個字都預測一個機率   
   
其中，每次都取最高機率作為結果的方式，叫做greedy decoding，也就是這個方法，使得我們預測的結果都是常見的用詞   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/6.png)   
   
另一個常用的做法，就是分別取最高機率的前幾項，丟到模型預測，預測結果再合併出一個列表，繼續取前幾項預測。這個方法則叫做beamsearch decoding   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/7.png)   
beamsearch之後，我們取第二項的結果，效果會有所改善，但效率太差，而且改善不明顯，還是會預測出差不多的事情。   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/8.png)   
   
有一個新的想法出現啦，在ICLR2020的《THE CURIOUS CASE OF NEURAL TEXT DeGENERATION》提出，鑑於bert/GPT這類預訓練模型已經很會生成文本，就算我們不是給他機率最高的結果，它也可以講出一番合理的話。   
因此，我們用抽樣來選擇詞語～   
當然啦，抽一下機率十分低的詞，就算是T5也救不回來，因此，這個抽樣還是要按照字的機率來抽，而且範圍也要有所限制，免得抽到一些不好的結果。   
這篇paper提供兩個抽樣的方法，一個是最高機率的前x項去抽樣，一個是按照機率累加，加到一定值之前的項去抽樣。   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/9.png)   
第一個方法是普通sampling，怕遇到情況說，詞表第一項機率是0.80，第二項機率是0.10... 這樣抽頭幾項，可能會將機率低的也抽進來。   
因此就有第二個方法，假設我們將threshold設成0.85.   
第一項機率是0.80<0.85,ok。   
第二項的機率會跟第一項相加，0.80+0.10 = 0.90 > 0.85，因此第二項及之後便不會考慮。這樣可以有效避免sampling會抽到機率低的字。   
   
由於這個方法基於抽樣，所以每一次預測的結果都不會一樣。導致可用空間不到。但剛好，我們這個艾伯特測夢剛好就需要這樣的東西啊！！   
我們用這個方法來decode，最終得到的效果還不錯，兼顧了效率和神棍程度。decode的結果有稍微不流暢，可能是模型太小的緣故，但這個不就更加有高人的感覺嗎XD   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/10.png)   
   
我把模型放到demo網站上讓大家試看看效果~   
[voidful.tech](https://voidful.tech)   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/adgp/11.png)   
   
有興趣自己做一個的話，我也把他包在NLP套件中   
資料下載與預處理：[NLPrep](https://github.com/voidful/NLPrep)      
模型訓練和調整：[TFkit](https://github.com/voidful/TFkit)      
模型部署與試用：[nlp2go](https://github.com/voidful/nlp2go)      
也順便推廣一下，這一個系列的套件是希望讓NLP的end2end流程變得簡單方便，最簡單可以3行code跑其中的模型，每一個部分都模組化，因此也很好去加入自己的資料和模型，也可以很方便的驗證和測試模型。目前只有我一個人維護和修改，因此難以做到十分完善，期待大家的加入～～～～   
   
預告一下，下一篇會講我怎麼優化 18MB的閱讀理解模型，使其F1提高25% (目前還在努力進一步提高)   
   
不說了，我去夜市擺攤去～   
   
   
   
