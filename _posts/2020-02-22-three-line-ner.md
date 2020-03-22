---                                     
title: 三行code部署一個 DistilBert-繁中-NER模型
categories:                                     
 - Implement                                     
tags: 實作 3line
---

這篇文章主要介紹怎麼用我們的nlp工具包訓練及部署一個ner模型，以及分享一下當中的心得。  
我們模型將會採用以下的設定:  
資料集 - Chinese-Literature-NER-RE-Dataset  
預訓練模型 - distilbert-base-multilingual-cased  
資料預處理 - [nlprep](https://github.com/voidful/NLPrep)  
模型訓練 - [tfkit](https://github.com/voidful/TFkit)  
演示及api - [nlp2go](https://github.com/voidful/nlp2go)  
colab - [3lineNER](https://colab.research.google.com/drive/1x5DLBQ6ufRUfi1PPmHcXtYqTl_9krRWz)

首先，安裝所需要的套件

    pip install nlprep tfkit nlp2go -U         


## 下載/預處理

資料集下載，並且轉成繁體

    nlprep --dataset clner --task tagRow --outdir ./clner_row --util s2t         

我們所採用的是Chinese-Literature-NER-RE-Dataset資料集  
這個資料集有分7個NER的tag，從下圖可見Time metric organization abstract 這幾類佔比很少，估計這幾類的表現都不會太好；此外，metric和time的資料也很接近，估計也容易混淆。後續可以考慮合併Metric和Time這兩類，用focal loss減緩資料不平衡帶來的問題。

![Chinese-Literature-NER-Dataset 統計](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset/raw/master/ner.png)



## 模型訓練

用distilbert訓練模型，由於很快overfilling，幾個epoch即可

    tfkit-train --batch 10 --epoch 3 --lr 5e-6 --train ./clner_row/train --valid ./clner_row/test --maxlen 512 --model tagRow --config distilbert-base-multilingual-cased        

Distilbert 相比bert少了40%的參數，快了兩倍多，有Bert 97%的準確度。很值得拿來嘗試一下效果，其中的multilingual版本更可以支援中文，讓我們可以在colab上面也可以很快訓練出來。

從訓練的log來看，模型很快就開始overfitting，對此的可以適當調小learning rate，去減緩這種情況。

    =========eval at epoch=1=========      
    model: checkpoints/1, Total Loss: 4.135023178349079      
    =========train at epoch=2=========      
    100it [00:31,  3.21it/s]step: 100, loss: 0.16292362678611633, total: 2416      
    200it [01:02,  3.25it/s]step: 200, loss: 0.1580470992409768, total: 2416      
    300it [01:33,  3.20it/s]step: 300, loss: 0.15292472354572675, total: 2416      

另一個比較有趣的現象，是testing 的loss蠻高的，但對於準確度反而卻上升了

                   precision    recall  f1-score   support      
        B_Abstract       0.68      0.56      0.61       124      
        B_Location       0.90      0.88      0.89       944      
          B_Metric       0.56      0.72      0.63       189      
    B_Organization       0.66      0.71      0.68       142      
          B_Person       0.98      0.97      0.98      1433      
        B_Physical       0.00      0.00      0.00         0      
           B_Thing       0.84      0.90      0.87      1154      
            B_Time       0.80      0.91      0.85       563      
        I_Abstract       0.65      0.60      0.62       108      
        I_Location       0.91      0.86      0.88       929      
          I_Metric       0.55      0.70      0.62       183      
    I_Organization       0.58      0.72      0.64       121      
          I_Person       0.97      0.94      0.95      1022      
        I_Physical       0.00      0.00      0.00         0      
           I_Thing       0.85      0.85      0.85      1044      
            I_Time       0.81      0.88      0.85       566      
                 O       1.00      1.00      1.00      1895      
          
         micro avg       0.88      0.90      0.89     10417      
         macro avg       0.69      0.72      0.70     10417      
      weighted avg       0.89      0.90      0.90     10417      
       samples avg       0.90      0.91      0.89     10417      

這件事情其實也是合理  
loss 所希望的是將一個tag的機率預測到 1  
accuracy 則是這個機率最大就可以了

假設一個例子：  
gold label 是 time  
predict - time 0.43 metric 0.33 …..  
time這一項距離1很遠，loss還蠻大的。但同時time這一項的機率最大，因此這是一個正確的預測，accuracy會是1。  
猜測這裡會發生這個情況，是因為資料不平衡假設time和metric難以分類所導致。


## 部署和試用

選出訓練效果最好的模型，部署restful api或cli

    nlp2go --model ./checkpoints/3.pt --predictor biotag --cli       

驗證是不是最好的模型，我們可以對每一個模型都算一下f1，看看效果如何

    tfkit-eval --model ./checkpoints/3.pt --metric classification --valid ./clner_row/validation      

之後用nlp2go去部署模型，cli模式可以讓我們在colab上嘗試不同的輸入。也可以不加cli參數，直接host一個restful api的server，可以用 get 或者 post來獲得我們的結果:

    nlp2go --model ./checkpoints/3.pt --path ner      


![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pbnkfz_1/img2)

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pbnkfz_1/img3)


