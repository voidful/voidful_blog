---
title: 數說神經網絡-釐清反向傳播
categories:
 - NeuralNetworkInMath
tags: 數說神經網絡
---

前言：
這是寫給對機器學習和神經網絡有初步認識，但覺得雲裏霧裏，想要深入瞭解的讀者
希望能用盡可能簡單卻不失深度的數學去釐清整個過程



![](http://i.stack.imgur.com/wx8HD.png)


在上圖中，我們想要用綫來劃分X和0的區域，人腦去想，這個十分直覺
但對於電腦來説，這一條劃分X和0的綫，需要用公式畫出來，如y=mx+c之類
但上圖的分類明顯不是畫直綫就可以解決
這種不能通過綫性組合解決的非綫性問題
目前有兩派的想法來解決這個問題

  - 嚴謹數學推導為代表的SVM 支持向量機
  - 以數據而言的 Neural Network 神經網絡

他們主要的想法是將整個空間投射到一個更加高的維度中，而在那個維度上數據則會是綫性可分
SVM對於這整個過程都有完整的解釋，但實現上並不好做
神經網絡則是換了個思路，通過try by error的方法，逼近想要的答案

到底是什麽意思呢，其實是這樣的：

![](http://i.stack.imgur.com/wx8HD.png)


給定一堆數據，且已經知道是屬於哪一類
初始化一堆方程組，劃分開輸入的數據
對比劃分結果和實際結果，按照差異來調整方程組的權重
目的是希望能根據實際數據去學習背後的分佈

在神經網絡中，會將這些方程組用圖表示，其中的一個方程就是圖中的節點，也叫做neuron 神經元

![](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-09-at-3-42-21-am.png)


我們可以調整的權重有 W 還有 B ，從綫性方程的角度來看，其實就是傾斜方向和原點的移動距離

![](https://images.slideplayer.com/26/8825804/slides/slide_4.jpg)


説不定現在你已經有疑惑了，目前所介紹的神經網絡都只是些綫性的方程組
怎麽可以讓處理非綫性的問題？
處理的做法是在每一個綫性方程輸出的時候，加一個activation function 激活函數
activation function將會是非綫性的函數，對綫性方程的輸出做非綫性轉換
來讓神經網絡得到處理非綫性的能力

但這個非綫性轉換的細節應該怎麽處理？這個問題繁雜得讓人頭大，所以我們都統統交由神經網絡處理，去不斷調整這些方程組的參數，這些足夠多參數，超級複雜的方程組能fitting到同樣複雜的非綫性分佈，讓神經網絡能得到絕佳的結果，而調整參數的過程，不需要我們參與，我們可以喝杯咖啡，吃個下午茶靜靜地等待就好

在此之前，還需要看下神經網絡是怎麽調整這些參數，如同之前所説的，try by error
我們一開始給定隨機參數得到的一個結果，跟我們實際結果對比
對比的方法我們叫做loss function，他們之間相差的量叫做loss，我們則希望最小化loss來得到最佳預測的結果
如果要最小化loss，取決於神經網絡預測的結果，也就取決於其中的參數w和b，不同的w和b加上非綫性轉換得到loss，其變化可以説是複雜且難以想象的。每一個對應的loss組成的綫，會是一個非綫性函數。
在多組w和b下則會變成增加loss的維度，不同的參數會帶來高低起伏的loss

![](https://i.stack.imgur.com/w7ARo.png)


對於這種曲綫複雜的圖形要fitting出它的函數，可以嘗試只看某個函數的一部分，這個部分的分佈我們可以用相對簡單的函數來fitting，然後叠加起來

![](https://d2mxuefqeaa7sj.cloudfront.net/s_421598583FF9213D32A09FF85DCA01AF95E1BAE5A92E736CDBAA6DB3CF999AAF_1538411482501_.png)


從這張圖可以看出，我們取的那一部分可以看作是不同階的微分，然後一直相加
這個背後其實是 泰勒公式 Taylor's Formula , 在處處可微的函數中可以用多項式來逼近
假設x是我們的loss ，f(x)可以得到loss對應的參數，根據泰勒公式，可以展開成

![f(x) = f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2!}(x-x_0)^2 + \cdots + \frac{f^{(n)}(x_0)}{n!}(x-x_0)^n + R_n(x)](http://quicklatex.com/cache3/ab/ql_779b417eb436856dce1705588d9f52ab_l3.png)


如果我們只取到一階導數(微分一次)，則會用直綫來fitting x，二階會用抛物綫……
考慮到微分越多越難算，所以一階導數反而是最佳選擇
只展開到一階導數我們可以得到:

![](http://quicklatex.com/cache3/9a/ql_ddf093998cf66c60f09106b703d4849a_l3.png)


效果就會像是這樣

![](https://pic2.zhimg.com/80/v2-484252e306fad9dc96dbdd034ba326f6_hd.jpg)


根據loss畫出的函數來看，若希望loss盡可能小，我們的參數會是在山谷的位置

![](http://adventuresinmachinelearning.com/wp-content/uploads/2017/03/Gradient-descent-300x156.jpg)

![](http://quicklatex.com/cache3/9a/ql_ddf093998cf66c60f09106b703d4849a_l3.png)


整個過程會是先找一個參數w作爲起始點，通過一階的泰勒方程組找到下一個w,一直找到w越來越小爲止

但我們怎麽能確保找到的w會讓loss一直變小呢？
f'(x_0)可以看作是對x fitting得到直綫的斜率(slope)
當斜率為正的時候，w越大，loss可以說是越往上走，反之亦然

![](https://d2mxuefqeaa7sj.cloudfront.net/s_421598583FF9213D32A09FF85DCA01AF95E1BAE5A92E736CDBAA6DB3CF999AAF_1538418025903_y+equals+x.bmp)


因此爲了確保能一直往最小的方向走，新的w的值，相比原來的值，應該會是在slope的反方向
我們會將slope乘h，控制每次更新所走的量，這個值不應該太大，因爲走越遠這個fitting就越不準確了
總結下來，我們的根據一階泰勒展開得到該點fitting的綫
根據這條綫的微分，可以得到要更新的方向，然後我們設每次更新的量值h，可以寫出這個公式

![x_n+1 = x_n + f'(x_n)*h](http://quicklatex.com/cache3/4f/ql_d4466496900b4c5ad585b369b229d74f_l3.png)


多維空間則會換個符號寫成

![{x}_{n+1} = {x}_n - h \nabla({x}_n)](http://quicklatex.com/cache3/a6/ql_a5a11276b7b0dc499c6176bd44392aa6_l3.png)


可以想象這樣一步步迭代的，最終就能畢竟極小點了，而這就是所謂的梯度下降，整個更新loss的過程也叫反向傳播

好像隱隱約約又有一個新的問題，感覺一旦數據很多，會很慢
這個神經網絡還需要需要性能上的優化

最直觀的從反向傳播入手，有兩個方向：

追溯到loss的計算，我們有一組參數w,b，想知道這組參數的loss是多少，我們需要將真實的數據代入去計算，而當訓練集很大，我們一筆一筆代入計算就意味著每次更新loss的成本十分高，而如果只是隨機選一筆數據算loss，又會不太準確。代入多少數據算loss，也是其中一個探討優化的話題之一，通常這個參數會叫做batch_size

而算loss的時候，控制我們每次更新的量值h，如果h越大走的便越遠，更加快走到極小點，但卻可能導致結果不準，那一開始的時候，由於距離極小點很遠，h可以設比較大，之後慢慢減少。這個h通常會叫做learning rate，有各種不同的優化方法，也是另一個優化的方向。

因此下一篇的目標，是談談batch size 和learning rate 的優化~

參考資料:  
https://www.quora.com/What-is-the-relation-between-a-Taylor-series-approximation-and-gradient-descent-algorithm  
https://www.jianshu.com/p/d66db9e56074  
https://zhuanlan.zhihu.com/p/36503663  
https://zhuanlan.zhihu.com/p/38541058  
http://sofasofa.io/tutorials/python_gradient_descent/  
https://www.matongxue.com/madocs/7.html  
https://www.zhihu.com/question/21149770  
http://cpmarkchang.logdown.com/posts/436316-optimization-method-newton  
https://blog.csdn.net/bitcarmanlee/article/details/52195617  
https://kexue.fm/archives/4277  
http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/  
