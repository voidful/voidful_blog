---
title: 論文解析:Real-time Personalization using Embeddings for Search Ranking at Airbnb
categories:
 - Paper Reading
tags: 論文解析
---

**2018 KDD Best Paper**

試想一下，我們在Airbnb上要預定房間，通常結果都是眼花繚亂，一個個點擊查看會花費大量時間成本。
爲了減少查找的麻煩，點選了一個房源後，提供相關的推薦，如 :
當前選擇的房源 : 30塊的海景房。
推薦列表可以有 其他便宜的海景房，或者 海景很好的中價位房。
推薦列表越貼合我的喜好，能讓我越快找到自己想要的房源。


傳統方法會是 定義 用戶的喜好,年齡,性別…… 作爲預測的特徵去做預測 ，這種做法常會遇到資料缺失，特徵不夠等各種問題。
這篇paper的想做到的，是建立一個模型去找到可能的特徵，之後做預測，撇除人工因素帶來的偏差和減少處理特徵的麻煩。
也就是說 - 用一個模型去描述 用戶所選的房源 之間的關係，再基於這個模型去做預測。

而這個模型，則是基於word2vec的想法演變而來。

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img1)  


word2vec 是基於Distributional hypothesis而來，Distributional hypothesis是說，相似的上下文中的詞，意思也會接近。所以word2vec就是 一個描述 詞 與 上下文 的相關去建立embedding。
爲什麽說是embedding呢，因爲在後續實驗發現，word2vec還學到了同義詞，反義詞等語義蘊含特徵，我們用word2vec取得的詞的數字化表示，也就是詞向量，裏面也蘊含的詞的語義訊息。
這一串詞向量在之後的task，比如 閱讀理解，文本分類，機器翻譯中作爲模型的輸入，可以讓模型有更加好的表現。

看回我們的目標，用一個模型去描述 用戶所選的房源 之間的關係，再基於這個模型去做預測。word2vec就正好能做到這件事情。但由於我們的輸入不是文本，因此對這個模型要有相應的修改。

我們拿到用戶訂房時，所點擊房源列表
取其中一個房源出來，使得模型預測其周圍 Windows size内(如圖是：m)房源的機率越高越好(會限制其範圍為0-1之間，因此越接近1越好)。
在所有房源中，這幾個房源被預測為1，按理説也要讓其他房源預測為0才準確。但這樣運輸量太大，因此只找幾個其他房源就好讓其預測為0就好(這個優化方法叫做negative sampling)

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img2)  

因此目標函數的公式可寫成

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img3)  

使用者從瀏覽房源到最後預定，可能隔了很久的時間，因此都放在同一個列表預測不太合理，所以就每三十分鐘内的點擊才算是在同一個序列中，三十分鐘外的則斷成兩個序列吧

我們也希望加强最後預定房間的權重，希望在每一次預測除了將兩旁的房源預測為1以外，也要將最終預定的房間也預測為1。爲了使得最終預約房源，就算不在windows size内也可以做上述運輸，因此就放在負樣本的時候做 ： 

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img4)  

還有一個可考慮的場景，我們訂房的搜尋都是在一個地區的範圍内，我們當前的選擇其實是排除這個地區其他的房間而來的，我們也希望模型能注意到這一點，因此把地區内的其他房源也當作負樣本：

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img5)  

這樣就完成建模，可以訓練出embedding，這個embedding在論文中叫 Listing Embeddings

當有一個新房源，就像是word2vec中的未登錄詞一樣，我們可以找幾個有相識特徵的房源，加起來平均來當作它的embedding

看一下這個embedding是否有學到房源之間的聯係

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img6)  

看來這個方法有成功抓到房源之間的聯係，背後也蘊含了一些特徵，如

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img7)  

通常租整個房源的人比較不傾向租公用房源，因此其與公用房源的相似度是最低的。
我們完成的特徵的建模，取得embedding，是不是可以做預測了？

但是這個方法還有一些缺陷，讓我們不得不停一停。

試想一下這樣的情景，我去日本玩的時候在Airbnb去找充滿和風的房源，這個模型有學到我對此的喜好，但一旦我去美洲旅行，這個模型很可能還是會給我推薦和風房源。
或者當我的瀏覽,預定的資料太少，也很難學到有價值的特徵。
瀏覽的房源大大多於預定的房源，資料很不平衡，但結果才是最重要的。
人是在不斷改變的，比如年齡越來越大，訂房的選擇其實也不同，而之前的模型並沒有處理這些情況

因此我們也希望讓embedding能學到以上的訊息。之前模型的輸入和輸出也要做相應改變了。

我們之前輸入的是房源的id，這個id是獨一無二的，換個角度想下，會選擇這個房源來看，其實是因爲這個房源有我們想要的，如：4人房，加州，有熱水和WiFi 這四個特點，我們可以用房間的特點來代替房間的id。
就會像是 ：原本是id - 1234 會變成 4人房_加州_有熱水_有WiFi，”4人房_加州_有熱水_有WiFi”這一串所代表就可能不止一個房間，而是這類型的房間。這個房源只有在這裏有，但這類型的房源在全世界有很多，用類型而不是id可以讓其更通用。

對於user我們也可以用user的特點來建立模型，user的id 也可以變成 男_18-30歲_講中文 這樣。
在論文中，房源的特點跟用戶的特點主要取這些項目：

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img8)  

特徵化后會得到這樣的結果：

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img9)  

我們可以對使用者所有的預定記錄建立一個這樣的序列：


![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img10)
  

第一個Ut 可以代表使用者在那時的user type，在這個user type有預定過Lt(i-m) 到 Lt(i) 這些特點的房源，之後usertype改變了(可能是年齡變大，訂房次數到一定的量之類的原因)，改變后的user type 有新的預定。這個方法可以很好表達使用者的改變和訂房的不同。
因此基於這樣的sequence，我們套回之前的框架建立模型：

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img11)
  

我們預定之餘，也可能有被拒絕的記錄，這些被拒的記錄可以當成是負樣本

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img12)
  

因此目標函數就是最大化以下兩個公式

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img13)  

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img14)  

這個方法在論文中叫：User-type & Listing-type Embeddings

我們有了這個embedding去提取特徵，我們就可以拿來做預測。
將embedding維度設小(論文是32)，讓其在實時搜尋中效率更高，之後用以下方法

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_rtpuefsraa/img15)
  

提取特徵，將這些特徵丟到模型裏面，做ranking。

整篇paper大概是講這樣的事情，實驗的細節，預測的詳情可以參考[``原文](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb)
  
