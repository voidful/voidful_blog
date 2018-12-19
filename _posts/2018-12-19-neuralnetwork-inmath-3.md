---
title: 數說神經網絡-激活函數
categories:
 - NeuralNetworkInMath
tags: 數說神經網絡
---

前言：
這是寫給對機器學習和神經網絡有初步認識，但覺得雲裏霧裏，想要深入瞭解的讀者  
希望能用盡可能簡單卻不失深度的數學去釐清整個過程  
---

## 引子
上文介紹了如何在 loss function 中找最優解(全局最低點)的各種方法  
我們細看每一個神經元，都會有一個相對應的loss  
而我們的output只有一個  
那output的loss傳到神經元呢？  
這個就涉及到反向傳播的細節：  
我們對於output的loss微分可以得到  
當前output的下降方向  
而output的值收到上一層的神經元影響  
那怎麽知道不同的神經元對於output loss影響有多大呢  
我們可以對其做偏微分來得到  
整個結果會如下圖  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_3/img1)   
x的梯度就是在一層層的偏微分中得到的  
這個過程也叫鏈式法則  
通過鏈式法則得到某個神經元的梯度  
再通過上一篇文章講到的[梯度下降優化法](https://voidful.github.io/voidful_blog/neuralnetworkinmath/2018/10/26/neuralnetwork-inmath-2/) 來得到最優參數  

##  問題在於激活函數  
到現在一切都很美好，但問題就出在這裏 - 爲了解決非綫性問題而加入的激活函數  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_3/img2)    
激活函數(activation function)的作用除了讓我們的神經網絡具備非綫性能力之外  
也可以壓縮輸出，或看作做normalization，因爲不同的壓縮方式，所以得到很多不一樣的激活函數  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_3/img3)    
在剛剛講的鏈式法則傳遞的過程中，也要對於激活函數做偏微分，結果在上圖  
做微分可以理解為得到激活函數的斜率嘛  
然後我們看看Sigmoid 這個激活函數  
Sigmoid 在越遠離y軸的時候，斜率變化其實十分小  
意味著梯度更新會十分緩慢，當一個網絡叠很深的時候  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_3/img4)  
sigmoid經過層層傳遞，梯度會越來越小，小到沒有了  
這樣參數不會更新下去，訓練也就停止  
這樣的狀況會叫做梯度消失  

##  每一個激活函數背後，都有一段故事  
除了梯度消失的問題外，Sigmoid 會將所有值轉到0-1之間  
如果輸入的數據有正負值，經過Sigmoid之後正負的訊息就被丟掉了  
爲了避免這個問題而有了tanh的誕生，相比sigmoid 能更好的適應不同的數據  

而對於梯度消失的問題，讓其輸入等於輸出就可以來保存梯度，而讓負值等於0就可以做到非綫性的效果，也就有了ReLu  

當然了，ReLu還是會遇到正負值的問題，所以就采取將負值壓縮到一個範圍的方式來解決，也就是leaky ReLu  

[不同點的斜率可視化](https://engmrk.com/wp-content/uploads/2018/05/ice_video_20180508-144902.webm)

## What's Next ?
至此，神經網絡的整個過程都walk through了一遍。  
走出新手村  
神經網絡裏面有各種不一樣的架構 CNN RNN Transformer ......  
也有很多事情的學習方法 - GAN,RL ......  
期待後續更新吧~  
希望這個系列的文章能帶來一些幫助。  
難免會有錯誤或不清楚的地方，歡迎勘誤~  

參考資料:   
http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf  
https://engmrk.com/activation-function-for-dnn/  