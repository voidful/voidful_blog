---                                        
title: 三行code部署一個 ALBERT中文閱讀理解問答模型
categories: 
 - Implement     
tags: 實作 3line
---   

三行code部署一個 ALBERT中文閱讀理解問答模型
安裝環境   
   
    pip install nlprep tfkit nlp2go -U   
   
直接上code～   
   
    nlprep --dataset drcdqa --task qa --outdir ./drcdqa/   
    tfkit-train --maxlen 512 --savedir ./drcd_qa_model/ --train ./drcdqa/train --valid ./drcdqa/test --model qa --config voidful/albert_chinese_small  --cache   
    nlp2go --model ./drcd_qa_model/3.pt --cli --predictor qa   
   
效果   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img1)   
   
   
train一個epoch只需要4分鐘   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img2)   


## 機器閱讀理解的問答模型？   
   
首先，你可以會有疑問說，什麼是機器閱讀理解？   
在這裡，我們姑且將其定義為:   
   
- 給定一篇文章   
- 提供一個跟文章相關的問題   
- 機器可以從中抽取出答案(答案在內文裡面)   
![圖1: 機器閱讀理解例子](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img3)   
   
   
如上圖例子，答案的來源便是文章中藍色的部分，我們也會叫這一類模型做抽取式閱讀理解模型。而在內文中找答案的好處在於，機器回答不會超出文章的範圍，得到難以預測的結果。   
   
![圖2: 從內文中找答案例子](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img4)   
   
   
另一個好處便在於，訓練比較容易和簡單，在小模型上，十分鐘左右就可以訓練到收斂！   
   
## 訓練的過程   
抽取式閱讀理解模型的，是訓練模型找答案在哪裡。   
   
![圖3: 抽取式閱讀理解對機器而言](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img5)   
   
   
當機器收到問題，便會從文本中找尋答案在哪裡，因此閱讀理解模型會有兩個輸出 - 答案開始位置和答案結束位置。但是，機器學習常見的問題不是分類和回歸，看似並不能預測位置啊？   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img6)   
   
   
解決方法很有趣，把輸入文本最大長度當成類別數量。如上圖，最大文本是32個字，就分32個類別，每個類別代表一個位置。這樣針對開始和結束位置預測兩次就可以得到答案區間！   
   
   
## 模型   
訓練方法有了，就差模型和資料～   
在繁體中文上，台達有放一個閱讀理解資料集：從2,108篇維基條目中整理出10,014篇段落，並從段落中標註出30,000多個問題。   
[台達閱讀理解資料集 Delta Reading Comprehension Dataset (DRCD)](https://github.com/DRCKnowledgeTeam/DRCD)   
   
模型的話，我們這次會用中文版的albert。albert是一個預訓練模型，背後是用transformer模型做self-supervised learning(將字遮起來，然後預測遮起來的字之類的)。而且，Albert還針對模型做了一系列輕量化：   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img7)   
   
   
這使得albert模型變得足夠小，GPU的ram可以少用很多，就可以把batch加大，加速訓練啦～   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img8)   
   
   
現在重新來看看怎麼把這個模型訓練出來，為了簡化這個過程，我做了幾個套件來幫忙：   
資料下載與預處理：[NLPrep](https://github.com/voidful/NLPrep)   
模型訓練和調整：[TFkit](https://github.com/voidful/TFkit)   
模型部署與試用：[nlp2go](https://github.com/voidful/nlp2go)   
   
使用這些套件來訓練albert閱讀理解模型的過程   
   
1. 下載資料集，轉成QA的格式   
    nlprep --dataset drcdqa --task qa --outdir ./drcdqa/   
2. 訓練模型，最大長度設512，預訓練模型是albert_chinese_small   
    tfkit-train --maxlen 512 --savedir ./drcd_qa_model/ --train ./drcdqa/train --valid ./drcdqa/test --model qa --config voidful/albert_chinese_small  --cache   
3. 拿最好的模型試看看，通常3個epoch左右就收斂了，--cli 可以讓你在terminal或者colab上面測試模型，也支持搭成restful api的形式   
    nlp2go --model ./drcd_qa_model/3.pt --cli --predictor qa   
   
效果嘛   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/tlaq/img1)   
   
   
