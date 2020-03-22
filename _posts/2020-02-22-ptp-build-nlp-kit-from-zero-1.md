---                                     
title: 預處理，訓練，發佈-從0開始打造全套NLP工具包 (1)總體架構
categories:                                     
 - Implement                                     
tags: 實作 NLPkitScratch
---                                     

transformer架構的模型最近在大放異彩，我們都想將不同的資料集和任務都換到transformer來試看看有什麼突破。麻煩在，嘗試不同的資料集和任務免不得有不少重複的操作，現在huggingface project過於臃腫，難以靈活地換到不同的資料集上。  
我們換個想法，將其中的部件拆分，變成幾個主要的部件各司其職。主要是希望降低耦合性，讓我們可以專注解決每一個階段的問題，同時也更好維護。這也是打造這樣一套nlp訓練工具的原因。  
一個完整的機器學習project，基本上都是 預處理 - 訓練 -預測 這樣的步驟。  
按照這樣的想法，這一套nlp工具包也會如此：
- 資料預處理 - [nlprep](https://github.com/voidful/NLPrep)
- 模型訓練 - [tfkit](https://github.com/voidful/TFkit)
- 演示及api - [nlp2go](https://github.com/voidful/nlp2go)

廢話不多說，首先看看這一套nlp工具包怎麼運行，以我們要訓練一個NER模型為例子：  
首先，安裝我們這三個工具包

    pip install nlprep tfkit nlp2go -U   

下載clner資料集，並且轉成繁體

    nlprep --dataset clner --task tagRow --outdir ./clner_row --util s2t   

用distilbert訓練clner模型，由於很快overfilling，幾個epoch即可

    tfkit-train --batch 10 --epoch 3 --lr 5e-6 --train ./clner_row/train --valid ./clner_row/test --maxlen 512 --model tagRow --config distilbert-base-multilingual-cased   

選出訓練效果最好的模型，部署restful api或cli

    nlp2go --model ./checkpoints/3.pt --cli   

是不是很簡單，三行code就可以完成整個過程。這一篇主要會介紹每一個工具設計的想法和流程，希望能給大家一個整體的畫面，在之後可以細講其中的設計，說不定能吸引更多人維護或者能得到寶貴的建議XD


## 資料預處理 - [nlprep](https://github.com/voidful/NLPrep)

nlprep的想法，是打造一個預處理資料的框架，讓

- 使用者： 一行code就可以準備好訓練資料
- 開發人員：簡單的修改就可以自己的資料集

對使用者而言，他們所關注的應該是之後機器學習的部分，希望能快速得到可用於模型訓練的資料，之中也能做到一些常見的預處理。  
也就是，使用者只要能提供以下的資訊，便希望直接得到能拿來訓練的資料集：

- 使用者想要處理的資料集
- 訓練的任務類型(分類/標記/生成…)
- 預處理(繁簡轉換/過濾/normalization…)

根據上面的要求，我們梳理一下預處理的過程：  
下載文件，轉成對應任務格式，過濾預處理，輸出  
而對於開發人員而言，則希望能在其中加入新的資料集，或者加入更多預處理的工具。  
開發人員要加入的文件，改動的地方應當要越少越好，才能盡可能避免出錯和迷惑。  
因此整個流程可以改成:

1. 抓取原始資料：下載/讀文件
2. 轉換成統一格式
3. 統一格式轉換成相對應的任務格式(分類/生成/標註)
4. 過濾/增量/清洗
5. 輸出

其中 2和4 部分 便是讓開發人員修改和發揮的空間。


## 模型訓練 - [TFkit](https://github.com/voidful/TFkit)

模型訓練是比較複雜的部分，避免從頭開始的困難，這一部分會利用huggingface的transformer project修改而成。

也是一樣，從使用者和開發人員兩個角色入手

- 使用者： 一行code訓練模型
- 開發人員：簡單的修改就可以加入自己的模型

使用者而言，訓練一個transformer架構的nlp模型，其實就是拿一個預訓練模型finetune的過程。因此，需要確定的是：

- 哪一個預訓練模型(bert/albert/t5/gpt)
- finetune的任務(分類/標記/生成)
- 訓練/測試資料

其餘還有一些模型訓練所要用到的參數

- learning rate
- batch
- epoch

還有很必要的事情，便是在訓練的過程中紀錄訓練和測試的loss，讓使用者好判斷模型的發展。再進一步，還能用各種指標來驗證模型的效果。


開發人員而言，常用到的部分是:

- 增加其他任務和模型
- 加入loss function
- 加入更多驗證的benchmark

這幾個部分實作起來並沒有想像中困難，pytorch本身的設計已經解決其中的問題。  
在pytorch上，一個deeplearning模型的訓練流程會是:

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pbnkfz_1/img1)


以model為例子，model一律接dataloader作為輸入，算出loss之後回傳，模型個中細節，交由開發人員調整設計。


## 演示及細節 - [nlp2go](https://github.com/voidful/nlp2go)

這一部分相對比較單純，主要是提供一個相對簡單的interface丟資料去測試模型的結果: 可以是restful的api，也可以是cli  
tfkit中模型的預測，主要目的是為了算loss，對於演示模型效果並不太合適，因此需要將tfkit預測的結果稍作轉換才能輸出。  
往後也可以加入簡單的前端和analytic，使得demo階段能得到更加清晰的反饋。  
最終的效果：

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pbnkfz_1/img2)

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pbnkfz_1/img3)

