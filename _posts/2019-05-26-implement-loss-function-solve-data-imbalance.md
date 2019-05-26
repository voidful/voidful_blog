---   
title: 實作 從loss入手，解決資料不平衡帶來的訓練問題   
categories:   
 - Implement   
tags: 實作   
---   
   
最近訓練網絡的過程中，由於訓練資料不平衡，導致結果不如理想。   
研究目前方法之後，對loss function做了一點點修改，減緩over fitting的情況。   

比較常見的情況之一，會是在做分類的時候，某一類的樣本總是特別多，這就導致模型會偏向將結果判斷成這一類，可以輕鬆得到很高的準確度，但這個準確度對實作來説，並沒有什麽用。   
   
常見的例子是：資料中有大量正樣本，比如 100筆資料中，當中有95筆資料是正樣本，5筆資料是負樣本。那模型只要將結果都預測為正樣本，模型準確度就可以有95%   
   
**神經網絡訓練跟樣本不平衡的關係**   
那麽從神經網絡的角度來看，可以怎麽解決這個問題呢？   
簡單總結下神經網絡訓練過程，就是根據我們的 objective function得到離目標結果有多近(loss/梯度)，通過反向傳播的方式讓預測結果與目標結果越近越好。   
   
當樣本很不平衡，如剛剛的例子，100筆資料中，當中有95筆資料是正樣本，5筆資料是負樣本舉例好了，若網絡將所有樣本預測成正樣本，所以正樣本的loss會是0，負樣本的loss則會是1，有五個負樣本，loss就有5。   
收到loss以後網絡開始調整參數，若干輪訓練后，每一個正樣本或多或少都會有一點loss，比如本來是100%確定是正樣本，現在變成99%確定了。這1%的差距也就是正樣本的loss，假設95個正樣本都掉了1%，正樣本提供的total loss就是0.95了。此時負樣本已經做很好了，單看負樣本loss也降到2。   
   
這時候發現，我們回傳總共的loss會是2.95，正樣本這些一點點，一點點的loss，積少成多以至於最終正樣本提供的loss佔總共的32% ！   
可是我們在乎99%到100%的差距嗎？顯然不是的，但這一點點的差距在大量樣本的帶動下，影響著loss的計算，使得網絡的更新不如人意。這也是從神經網絡的訓練來看，樣本不平衡會帶來的問題。   
   
**對做很好的地方減少關注 - Focal loss**   
人總會過於關注自己做得好的地方，看來神經網絡也一樣XD   
對於預測來説，51%是正樣本，跟81甚至99%其實沒有差別。最後預測出來的結果也是正樣本。   
預測結果越接近正確的結果，其實也就越不需要更新了。但完全沒有梯度也不好，其可能受其他資料更新的結果影響，使得結果不進反退   
   
爲了讓結果能保持下去，我們需要讓模型每次都要有梯度更新，而準確度越高，更新的量應該越少，才可以避免被大量容易處理的資料帶歪。   
   
做法也呼之欲出了，給梯度一個權重，這個權重使得準確度越高，梯度越小，作為懲罰   
本來cross entropy的計算是   
$$ -log(Px) $$    
越高的機率，意味越不需要更新，因此懲罰越多，因此可以改成   
$$ -(1-Px)* log(Px)$$    
為了讓這個懲罰更加有力，可以讓其變成指數級   
$$ -(1-Px)* log(Px) $$    
這個就是focal loss，如果在二分類，還會有一個調節權重的參數。  
Focal Loss的圖（關注0-1的部分就好）   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_lfsdi/img1)

   
**也要接受怎麼樣也做不好的地方**   
focal loss之後，有研究繼續往這個方向探討   
AAAI 2019 - Gradient Harmonized Single-stage Detector   
   
除了會存在大量很好分類的樣本，還會有一些怎麼樣都做不太好的樣本，強行去fit這些樣本，很可能導致網路over fitting   
因此，我們在網路做很爛的地方，也可以給他一個懲罰   
   
這篇paper也想進一步，讓這個懲罰參數，可以根據當前情況動態設定   
比如這個梯度出現次數很多的話，可以讓梯度除出現次數，這樣就可以根據樣本量動態懲罰   
   
想法比較簡單，但實作起來會比較麻煩   
每次計算的梯度總會或多或少不一樣，一段一段離散地分佈，使得樣本量估算會變很小，懲罰的力道就遠遠不夠！   
   
訓練時我們不可能一次統計所有樣本的梯度，免不了要切成batch。在處理一個batch的時候還沒能知道整體情況，怎麼做到動態懲罰呢？   
   
要解決這兩個問題，要預先設定好梯度區間，落在同一個區間都會視為這一個梯度的樣本量。而統計梯度，則可以採用移動平均來近似。   
   
說真的，相對focal只改一行code，這個實作起來還是有夠麻煩，那有沒有結合以上兩個優點的方法呢！   
   
**兩全其美的方法？**   
   
換言之，我們希望尋找一個兼顧以上兩個paper的方法，設計一個可以對於準確度很高以及很低兩種情況給予懲罰，使得網路不好被這些樣本帶歪。   
   
中間高，兩頭低，是不是剛好想到一個很符合這個要求的 - 高斯曲線   
   
![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Normal_distribution_pdf.png/1024px-Normal_distribution_pdf.png)   
   
下一個問題，就是參數應該怎麼設定了   
高斯曲線有兩個參數，中心點和開合程度   
   
中心點可以設定在0.5 在這個不上不下的時候權重應該最高   
   
開合程度嘛，可以參考focal loss的曲線設定，我們應該讓最低跟最高點落差明顯，而同時在0和1之間不能跌到負數，不然權重就亂了   
   
同時，對於做不好的結果，懲罰也不應該比做很好一樣。一來做不好的樣本佔比很少，而來一開始訓練的時候很多樣本也是做不太好，懲罰太大，導致收斂會變慢   
   
結合這三點 可以得出   
   
$$ \left(e^{\frac{-\left(x-m\right)^2}{2s^2}}\right)-0.1x $$   
我們可以稱之位：gaussian weight loss，簡稱***GW Loss***   
其對應的圖會是這樣：   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_lfsdi/img2)  
與Focal Loss對比，則會是這樣（focal loss為黑色，GW loss為紅色）
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_lfsdi/img3)    
以下是Focal Loss 和 GW Loss 的Code
```python
##Focal Loss

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean()
```   
   
```python
##GW Loss

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GWLoss(nn.Module):
    def __init__(self):
        super(GWLoss, self).__init__()
  
    def gaussian(self,x,mean=0.5,variance=0.25):
      for i,v in enumerate(x.data):
        x[i] = math.exp(-(v-mean)**2/(2.0*variance**2))
      return x
    
    def forward(self, input, target):
      
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  
            input = input.transpose(1,2)    
            input = input.contiguous().view(-1,input.size(2))  
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 *  (self.gaussian(pt,variance=0.1*math.exp(1),mean=0.5)-0.1*pt) * logpt
        return loss.mean()
```   
   
在pytorch的官方demo code實驗者個loss function如何   
   
[pytorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)     
   
[Colab](https://drive.google.com/file/d/1VYTRHcX_Zs2HPGIQ5SDwUXemZVUOXaZ-/view?usp=sharing)    
   
   
這個實驗的輸入是某個語言的字符   
判斷這個字符屬於哪一個語言，作為輸出   
   
從confusion matrix 可以看到，在英文法文荷蘭文中機率都很接近，不好分別   
而用了balance loss以後，他們變得更加好分便了   
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_lfsdi/img4)    
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_lfsdi/img5)    
   
![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/imp_lfsdi/img6)   
    
    
最後要說的是，loss function的選擇還是要根據資料的情況來決定，沒有絕對的方法xd   
      
參考自:    
Focal Loss for Dense Object Detection    
AAAI 2019 - Gradient Harmonized Single-stage Detector   
   
   
   
   
