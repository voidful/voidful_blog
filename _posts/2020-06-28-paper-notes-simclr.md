---
title: Self-supervise learning的新方向 - 從representation中下手(SimCLR,SimCLRv2,BYOL)
categories:
 - Paper Reading
tags: 論文解析
---

SimCLR - A Simple Framework for Contrastive Learning of Visual Representations   
SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners   
BYOL - Bootstrap Your Own Latent A New Approach to Self-Supervised Learning   
本文會從以上三篇論文出發,介紹下最近self-supervise learning的一個新方向。  


預訓練的目的是希望從大量的資料中學到有用的特徵，讓下游任務可以參考，從而得到更好的結果。常見的方式有 - 生成式(generative) 和 判別式(discriminative)。 ![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/YaycOTg.png)

生成式(generative)通常會將輸入壓縮輸入，希望輸出可以盡可能將壓縮的輸入還原。比如將輸入的大小降低一半，模型需要將其還原到原本的樣子。而還原的圖片每一個像素都要接近原來的樣子，這是很困難的事情。這導致
- 這類型的預訓練方法不好學(不好收斂)
- 模型也花很多心思在微調一個個像素，像素的差異對下游任務幫助並不大，模型會學到一些不那麼通用的特徵  

判別式(discriminative)則是會判斷某個輸入的類別。如輸入一張圖片，判斷這種圖片裡面有什麼。判別式模型訓練相對生成式簡單，但很受限於資料，導致  
- 需要大量標記好的資料才可以發揮效果。  
- 資料集的偏差會使得模型有偏差，這種偏差也會直接帶動下游任務中。  
- 模型從輸入資料的種類得到的特徵，未必對有新資料的下游任務起到幫助。   

這兩個模型都或多或少存在一些問題，導致預訓練模型學到的特徵不夠通用。  那麼有沒有方法，不需要大量的標記資料，同時學到的特徵更加通用呢？  
這裡介紹的三篇論文，就是往這個方向著手。  

首先，我們看看如何解決需要大量標記資料做預訓練的問題!  
近來最出名的解決方法，就是自監督式學習(self-supervise learning)啦。self-supervise learning的常見的訓練方式，會透過將輸入的部分訊息遮掉，然後預測遮掉的部分，使得模型能從大量的資料中學到有用的特徵。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/Bb0j8na.png)  

最近nlp界很熱門的BERT便是使用這種方法訓練。遮住文本中某個字，然後將這個遮住的字預測出來。沒有遮住的文本就可以作為預測線索，讓預測字能保有一定難度，也不會到太簡單。  

影像卻很難這樣做，就算是預測圖片的一部分，也是需要像素級別的調整，才能降低loss。很容易又陷入到特徵不通用的問題。`A Simple Framework for Contrastive Learning of Visual Representations` 就提出了一個很深意的解法 - Contrastive Learning。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/Eip9pGT.png)

Contrastive Learning的作法，是判斷兩個輸入是一樣還是不一樣。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/p25rkkF.png)


例如：輸入是一隻貓A的圖片，我們將右邊的圖片丟到模型逐一判斷，其中貓A的圖片應該判斷成一樣，另外一張貓B的圖片則要要判別為不一樣，而其他動物的圖片也要被判別成不一樣。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/15lD7BS.png)

在這個過程，模型所要做的，就是將輸入的圖片轉成向量，輸入到比對相似度的網路，然後得到兩張圖片的相似度的值，多會normalize到0-1之間，同一張圖片會越接近1，反之則會接近0。

清楚了整個過程，我們來分析一下Contrastive Learning的好處：
- 不需要大量標記的資料就可以訓練
- 任務簡單好學，容易收斂

但會不會就是因為任務太好學，導致學不到什麼通用的特徵呢！這是十分有可能的事情！因此，我們需要加大難度。加大難度有兩個方向，一個是讓比對的樣本跟輸入很像，比如從都是貓的圖片中找到同一隻貓，但這樣就需要找到同類型的照片來訓練，也就需要標記資料的幫助，有違我們的初衷。所以轉而投向第二個方向-加入噪聲，使得任務更加難學。  

SimCLR就提出兩個加入噪聲的方法，而BYOL則是在SimCLR之上再加入一個噪聲。噪聲越多會使得任務的難度提高，迫使神經網路不要用簡單的方法來判斷，使得它學出來的特徵更加通用。  

舉個例子來看，可能神經網路是根據有沒有毛茸茸的耳朵來判斷貓的，所有毛茸茸耳朵的生物就會被判斷成貓，這就不是一個好的特徵。當我們加入噪聲訓練，把圖片裁切掉，使得某些訓練圖片沒有毛茸茸的耳朵，就會迫使神經網路找更好特徵去判斷貓，如瞳孔，臉部組成等，這些特徵相比用耳朵來判斷更加通用，也能應付不同的任務。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/EVoQ0DL.png)  

縱覽全局，我們希望做一個通用的預訓練模型，通過自監督式學習來用Contrastive Learning學到通用的特徵，為了避免太簡單。因此加入一些噪聲，使得任務更加難學。  

我們從提出這個想法的SimCLR入手，看看細節到底如何。  

## SimCLR - A Simple Framework for Contrastive Learning of Visual Representations   
這裡是論文的架構圖，首先會通過資料增量，將一張的變成兩個不太一樣的樣子(加入噪聲)，之後轉換成向量，再投影到另一個向量空間中(加入噪聲)，再從這個向量空間中計算兩張圖片的相似度。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/zccWSac.png)

我們將這個過程分成三個部分來分析 - 資料增量/投射向量/相似度計算  

### 資料增量  

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/JPVpKaC.png)
透過以上的各種方式，將a的原圖變成各個不同的樣子，有針對角度的旋轉，針對圖片內容的裁剪和模糊，針對顏色的濾鏡。這是為了**減少圖片的特徵**，迫使模型尋找更加通用的特徵。  

至於探討說哪一個特徵最有效，論文作者的方式簡單粗暴，用不同的資料增量法學出不同的預訓練模型，再用ImageNet驗證準確度的高低，越高代表模型學出來的特徵越有用。從上圖可以發現，單獨用某一種增量的策略並不能得到最好的結果，兩種增量方式的結合效果會更加好，也進一步印證難度加大，特徵會越通用。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/Nm3VW5W.png)


相比用原圖和增量圖去學是否同一張圖片，我們對一張圖片分別做兩種不一樣的資料增量，再判斷這兩種經過資料增量的圖片是否為同一張，會更加有挑戰性。因此論文便採用這樣的方式，結果資料增量就可以得到得到兩張不太一樣的圖片，再繼續訓練，輸入的圖片數量也會變成兩倍。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/gXtU7A2.png)



### 投射向量
資料增量以後呢，就需要將圖片轉成向量，這裡用的是ResNet-50，將圖片轉換成2048維度的向量。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/Mc7R6h9.png)

理論上，在這以後便可以計算兩張圖片的相似度，但論文卻再經過一層向量投射，將2048維度的向量投射到另一個向量空間中，為什麼要這樣做呢？  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/x09K9Cp.png)


在向量投射的過程中，圖片的一些特徵會被打散，使得模型更難判斷。這一點從上圖的t-SNE視覺化也可以看到，各個類別的圖片本了各自佔據一個地方，經過投射之後，各個類別都混在一起，難以分辨呢。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/C0beEJI.png)


從結果來做實驗，用原本的向量和經過投射的向量預測圖片有沒有經過 灰階處理/旋轉處理/濾鏡處理 之類，都發現經過投射的向量更難分辨出圖片的處理方式。這樣加入噪聲，也進一步迫使圖片學到通用表達方式。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/8fRF5i1.png)  


處理的細節來看，維度大小對於結果的影響不大，我們可以放心縮小投射到的維度。另外，非線性的投射會比線性來得要好。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/9jT8kPn.png)

### 相似度計算
得到輸入圖片的向量後，我們希望**同一張圖片會越接近1，反之則會接近0**。因此， 增量圖片A1 的正樣本會是 增量圖片A2，訓練使得其相似度越接近1越好。但這並不代表 增量圖片A1 跟 增量圖片B1，增量圖片B2 都不像。因此，還需引入負樣本，讓增量圖片A1 更除了增量圖片增量圖片A2之外都圖片，相似度都是0。  


我們便需要兩兩組合間都算相似度，以得到上圖的結果。顏色越深代表相似度越高，而同類別的圖片也應學到一定的相似度，不同類型圖片的相似度也要很低。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/J5lANrW.png)


但這樣做，運算量大得驚人。為了減少運算量，我們折衷不讓所有的圖片都成為負樣本，而是讓同一個batch都圖片成為負樣本。batch size是2的話，loss function就會像上圖那樣，分子的組合的機率都要比分母的其他組合要高，而且越接近1越好。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/PQB2l3y.png)  


上面的訓練，讓 增量圖片A1 跟 增量圖片A2 最近，但並不代表增量圖片A2就會跟 增量圖片A1 最接近，因此也需要針對 增量圖片A2 算一個loss。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/4GxsbvX.png)


加上batch內的其他圖片和他的組合，整個loss function就會是這個樣子。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/EixY9my.png)


那麼負樣本取決於batch的大小，batch的大小會不會很影響到最終到效果呢？實驗結果告訴我們，會！但影響沒有到特別大，隨著訓練步數的提升，差距會漸漸拉小。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/F62X2BS.png)

## BYOL - Bootstrap Your Own Latent A New Approach to Self-Supervised Learning  

剛剛也發現，我們需要大量的負樣本或者訓練的時間，才可以讓SimCLR可以做到很好的效果。如果負樣本剛好抽到的都是不同類型的物件，如從貓跟家具中找出貓，應該會簡單而且也很難學到什麼。  
因此，`Bootstrap Your Own Latent A New Approach to Self-Supervised Learning`這一篇呢，就是要把負樣本丟掉，不用負樣本也可以做到同樣甚至更好的效果。而方法的出發點也是**加噪聲**。  
BYOL跟SimCLR一樣，也是有 資料增量 和 向量投射 兩個噪聲，架構上還是一樣的。不一樣的地方開始於SimCLR相似度計算！由於把負樣本去掉了，之前相似度計算的目標也不一樣了。這裡改而用投射向量預測另一個投射向量。這樣可以避免生成式在像素調整上花過多功夫的問題，模型也對 原圖 和 資料增量 有所了解才可以預測出來。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/7Cr1By6.png)

理想很美好，但實驗結果卻發現，這樣會導致分數劇烈下降。這很可能是因為，負樣本也是一個噪聲，幫助模型去理解特徵之間的關係 - 就如同人和貓都有眼睛和鼻子這兩個特徵，模型透過有沒有眼睛跟鼻子來判斷人和貓，但不能分清楚兩者的區別。負樣本讓模型知道特徵擺放的位置，大小等不一樣的地方，來得到人和貓的區別。缺乏負樣本，學到的特徵不夠通用。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/mApNANc.png)

這是因為沒有學到特徵擺放位置，大小而導致的，因此我們需要加入一個噪聲，讓模型去學這部分的訊息。也因為這個噪聲是針對特徵而做的，作法也是直接在向量中間中，加一個轉換，使得特徵擺放的位置被打亂，要還原的時候，也需要多考慮這一部分。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/u7Jipcx.png)

但是這個Noise不好控制，太強了模型學不到任何東西，太弱了效果又不大，也不能太突兀，否則訓練也很難穩定下來。以訓練的穩定優先來看，用之前的weight是一個不錯的選擇，噪聲的大小就可以取決於拿多少之前的weight來用。論文就用指數移動平均來調節，越接近現在的weight影響越大。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/JX89uLg.png)

我們也沒有必要更新加了noise的網路，這樣會讓模型的收斂不穩定，所以最終的黑色框起來的部分，其實會是紅色部分的指數移動平均，並不會參與到網路的更新中。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/oB7XBJE.png)

最終的效果來看，減少對負樣本的依賴，相比SimCLR可以在更小的batch也能保持準確度。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/L4UpOQj.png)

而不管是SimCLR或者是BYOL，他們最終在ImageNet驗證準確度的結果，都可以看到他們超出了所有無監督的方法，甚至BYOL在400M的參數上，無監督的結果還十分逼近有監督的效果。這樣也證明以上兩個方法學出來的representation是十分有效而且通用的。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/Jt3H3jO.png)


## SimCLRv2 - Big Self-Supervised Models are Strong Semi-Supervised Learners  
SimCLRv2 和 BYOL 應該是兩個同時在進行的工作，SimCLRv2在改進SimCLR的方法也是簡單暴力，加大圖片Encoding的網路，以及多做幾層投射向量。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/sct1aSo.png)

接下來是如何將學到的特徵應用的下游任務中。一般的作法就是直接在下游任務上fine-tune嘛，雖然效果會有，但整個模型實在太大，特別是SimCLRv2直接將ResNet50換成ResNet152，不是什麼機器都可以跑起來的。因此，我們希望既能有到大模型預訓練的好處，讓模型在少量資料的下游任務上有不錯的結果，也能讓下游任務的模型縮小。  

模型經過SimCLR的預訓練後，會學到圖片的特徵。我們用這個特徵向量在少量有標籤的資料上去fine-tune，例如做車牌辨識。然後，我們讓模型再對所有unlabeled data做一次預測(包含預訓練和沒有標記的車牌圖片)。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/RjtBQOw.png)

我們再拿一個小一點的模型，將沒有標籤和有標籤的資料都丟去預測，沒有標籤的資料就以大模型的預測結果作為標籤來用。透過這樣的方式把大模型所學到的特徵**蒸餾**到小模型中。這個做法叫做知識蒸餾，大模型通常會叫teacher model，小模型則會叫student model。  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/zH7hAxl.png)

論文還發現，我們不一定要用第0層作為fine-tune和teacher，我們也可以考慮經過投射的那幾層  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/tRo8f4T.png)

在少資料的情況下，用投射層作為fine-tune和teacher效果更好  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/0e4h7wo.png)

有趣的是，就算是student的大小跟teacher一樣(student少了投射層)，做蒸餾還是可以提高效果！？自己對自己蒸餾也會有效果，這點十分有趣。猜測是由於蒸餾的loss會讓模型不那麼容易overfitting，從本來要將某個label的機率推到1可能變成到0.9就好。當然啦，這種做法的teacher跟student也不能完全一樣，不然預測準確度永遠都是100%，就學不到任何東西了。  

最後的結果，模型只用了1%和10%的資料就可以大幅到超遠SOTA。但相比前兩篇，這篇論文的創新之處好像不太夠啊XD  
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_simclr/eNNsAEm.png)


最後的最後，本文介紹了最近三篇預訓練的新想法的論文。新的風向已經不是在資料上做self-supervised training，而是改為從向量空間中入手，透過線性轉換，資料增量來增加資料的噪聲，提高預訓練的效率。可惜這三篇都是用在CV上，如果是NLP的話，會怎麼樣呢？  

參考資料:  
https://amitness.com/2020/03/illustrated-simclr/  
(很多圖片都是從這裡來的，這裡用視覺化的方式將SimCLR講得十分清楚)  
