---
title: 數說神經網絡-梯度下降的優化
categories:
 - NeuralNetworkInMath
tags: 數說神經網絡
---

前言：
這是寫給對機器學習和神經網絡有初步認識，但覺得雲裏霧裏，想要深入瞭解的讀者  
希望能用盡可能簡單卻不失深度的數學去釐清整個過程  
---

## 引子
上一次説到，我們反向傳播是爲了探索出loss最低的參數  
在圖上來看，就會像下圖的紅色路徑  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img1)   
那怎麽知道已經走到最低呢， 根據反向傳播的公式  
$$ {x}_{n} = {x}_{n-1} - h \nabla({x}_{n-1}) $$   
其實我們是透過一次微分，也就是斜率來判斷是否走到最低點  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img2)   
但明顯的，通過斜率的變化，充其量也只能知道這一點是在一個山谷下的最低點，而整張圖有多少個這樣的山谷？
在多個參數的神經網絡下，這樣的山谷其實多不勝數，每一個山谷都會有最低點，但整張圖的最低點只會有一個啊  
所以反向傳播的時候就要注意別被困住這些極小點之中
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img3)     
因此之後的内容，就是介紹各個想辦法脫離極小點的方法  

## 隨機，batch，簡單有效但不永逸的方法
輸入經過神經網絡之後會算出一個結果，那個結果和我們的預期有差異，再根據根據差異調整參數  
而這個差異往往是代入所有數據的整體誤差    
所謂考慮越多，鉗制更多。好比要讓所有人滿意的方法，通常會比讓一兩個人滿意的方法少很多  
爲了在所有數據的loss都可以下降，每一個參數往下走(loss更低)的路也只有一條，不管困難重重也只能走下去  
這就導致不時卡在極小點裏面的原因之一  
如果只考慮一部分數據而不是全部，其實也可以得到loss的大致趨勢，雖然不夠準確，但也足以往下降  
但若只用一筆數據，算出來的loss跟真正所有數據的loss天差地遠，按照這個調整的權重，也就不很好處理所有的數據  
這就好比只考慮讓一個人滿意的做法，通常不會讓所有人滿意  
而考慮一部分人的想法，可能因爲其他人也正有此意，比較能得到大家滿意  
所以選一部分數據算loss，之後調整權重，其實也等同讓整體loss下降  
那每次loss更新隨機代入一批數據，loss往下走的方向因而不太一樣(有些人比較滿意，有些人沒那麽滿意，有些人可能不太滿意)，讓loss的走向飄忽不定   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img4)    
每次隨機考慮一批數據這個就是其中一個優化方法，簡單而直接  
而且還可以大幅度減少RAM的負擔，在大數據的時代把所有數據都加載進來，RAM很可能就不夠用了，一批一批的加載，負擔就輕很多  
但選的數據量不能太少，太多的話占用的資源又很多，這個就要根據實際情況設置了
這個方法叫做 隨機梯度下降法(SGD)  
每次選多少數據通常會叫做batch_size，也是神經網絡中很常見的參數  

## 每一步走多遠，是一個值得深思的問題
雖然說隨機的加入，讓梯度下降時沒那麽容易陷入到local minimum，但還不夠保險，多找一個方法，多一份保險嘛  
從另一個角度入手，當我們的learning rate設置得足夠大的時候，其實也可以跳出local minimum  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img5)    
通過learning rate的調整，可以跳出local minimum的同時也可以讓loss下降得更快  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img6)
但太大的learning rate也可能難在minimum附近跳來跳去，反而到不了minimum  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img7)
因此learning rate調整的過程應該要動態，直觀的接近minimum的時候越小效果會越好  
根據反向傳播的公式  
$$ {x}_{n} = {x}_{n-1} - h \nabla({x}_{n-1}) $$   
我們可以讓learning rate每次更新都乘以0.9，簡單方便讓更新從慢到快  
但現實總是殘酷的，有可能過早就變得太小，loss就不動，結束訓練  
我們最終目的是希望加速梯度下降的過程，因此我們換個做法，加入新的一項   
就像加入一個外力，在一開始的時候用力去推，之後用的力越來越小， 讓其慢慢走到最小值  
這一項不能太大和太小，也要跟當前情況來設定，那就直接用上一次的值就好  
再加上 上一次移動的量 並每一次都乘以一個小於1的權重，讓其一直減少  
$$ {x}_{n} = {x}_{n-1} - h \nabla({x}_{n-1}) + m({x}_{n-1}-{x}_{n-2}) $$   
還有一個更加快的想法，我們加了一個推力讓他走得更快一些，我們也可以預想到這個推力會按照原來的方向再走遠一些，因此算斜率的時候我們就直接用加了推力之後的位置來算，可以走得更加激進一些  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img8)
$$ {x}_{n} = {x}_{n-1} - h \nabla({x}_{n-1}-{x}_{n-2}) + m({x}_{n-1}-{x}_{n-2}) $$   
這個加入外力的方法會叫做 Momentum  
連斜率都考慮進來的更快版本則是 Nesterov Momentum  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img9)
但跑起來還是不夠快，這是因爲在實際應用中，我們不會只有一組參數，我們參數有很多，每一組參數的分佈是不一樣的，應該一一而論，根據每一組參數的情況更新  
每一維更新的量應該也要參考其他維，如果這一維的參數比其他維度的斜率更加高  
則可以更新得多一點，反之則更新少一點嘛  
要得到 一個綜合所有維度的梯度，我們單看斜率是不夠的  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img10)
做二次微分才知道多個參數下更加具體的趨勢  
而有一個想法是用均方根近似二次微分的結果  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img11)
也因此用均方根來衡量所有維度的梯度平均值  
具體會是對每一個維度做偏微分，乘以learning rate之餘，讓其除以所有梯度的平方和，來借此放大縮小更新的值   
```
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + le-7)
```
但這個做法其實會遇到跟之前 一直乘0.9方法 一樣的問題，會讓更新越來越慢導致過早停止  
我們就像mountain一樣用前一次梯度的均方根作爲新一項加上去，同時也要保留這一項的偏微分，不然就沒有一一而論的效果了，因此可以將它們按比例取值  
```
cache += decay_rate*cache + (1-decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + le-7)
```
這個一一而論的方法叫做 - AdaGrad  
其改進版本叫做 - RMSprop 或 AdaDelta (這兩個方法本質上是一樣的,只是細節上略微不同)  

Momentum的提出讓梯度可以多走一步  
而adaGrad可以讓梯度按照不同維度做不同調整  
這兩個優化方法是從不同角度入手的，將他們合二爲一也是沒問題的~  
既要讓其走一步，也要避免更新愈來愈慢的問題，因此我們加入  
```
beta1 * m + (1-beta1) * dx  
```
還要根據每一個維度的情況來處理，要
```
/ (np.sqrt(dx**2)) + le-7)
```
其中dx**2部分，也要考慮到更新愈來愈慢的問題，因此改成  
```
beta2 * v  + (1-beta2) * (dx**2)
```
最終得到  

```
m = beta1 * m + (1-beta1) * dx
v  = beta2 * v  + (1-beta2) * (dx**2)
x += - learning_rate * m / (np.sqrt(v)) + le-7)
```
這個博多家之長的方法是 Adam  

最後，來一個 各種方法的可視化對比  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img12)
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/nninmath_2/img13)

## What's Next ?
我們算好了loss,將其回傳到神經元，那麽具體是怎麽將loss跟某個神經元的權重聯係上的呢？  
反向傳播的具體過程其實到現在都還沒有介紹  
其中還有我們的激活函數，很神奇的讓神經網絡變得可以處理非綫性問題  
但也帶來新的問題，比如説常常聽到的梯度爆炸和梯度消失  
下一個部分，將會從反向傳播的過程來探討激活函數  