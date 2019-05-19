---
title: Beam search之後，讓文本生成更加靈動的解碼方法
categories:
 - Paper Reading
tags: 論文解析
---

來自Allen AI 的 The Curious Case of Neural Text Degeneration   
來自Huggingface 的 transfer-learning-conv-ai   

在神經網絡下，文本生成的decoding，一般有如下兩種方法：   

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_tccontd/img1)


其中，Greedy 方法會每次選最高機率的一項作爲輸出，直到遇到結束符號，將這些輸出的機率相乘，得到該sequence的機率。   

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_tccontd/img2)


這個方法會遇到這樣的問題：   
第一次decode中，如果選第二項，最後sequence的機率比選第一項高   
也就是greedy decoding會miss一些更加可能的結果   

爲了考慮更多不同路徑，以找到最佳結果，同時也要兼顧效能，就有了Beam search   

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_tccontd/img3)


Beam search會有一個beam size，代表每次會搜尋多少條路徑   
比如上圖的例子，beam size爲 2   
第一步挑出最佳的兩個結果AB   
找到AB之下，下一步會出現的結果: AA, AB, BA, BB    
在這4個結果中挑選機率最高的兩個: AB,BB   
從AB和BB出發，列出下一步會出現的結果：ABA, ABB, BBA, BBB    
再選擇最後機率的兩個，重複直到結束符號   

可見，beam size的越大，就需要更多效能   
除此之外，由於beam search的機率是一個連續相乘的結果，越早遇到結束符號，所得到的機率會越大吧   
也就是說，beam search對於長度很敏感，長文本相對短文本會更少出現   


![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_tccontd/img4)


爲了解決這個問題，   

第一個方法：   
對機率取log，使得乘法變成加法   

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_tccontd/img5)


最終結果還可以除於長度   
這些技巧可以使得長度差距不那麽容易被拉開   

第二個方法：   
來源於：   
Correcting Length Bias in Neural Machine Translation    
Breaking the Beam Search Curse: A Study of (Re-)Scoring Methods and Stopping Criteria for Neural Machine Translation   
這個方法常用於機器翻譯，先預測出輸出結果的長度，再做beam search   

但是在一些比較靈活的文本生成上，比如問答機器人，按圖說故事，寫新聞等，難以確定輸出文本的長度，也有研究發現，beam search的輸出，與人正在講話的機率分佈，很不一樣   

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_tccontd/img6)


Beam Search 在現在看來，好像不是創作型文本生成的最佳方法，那還有沒有其他思路呢 ？   
再次分析一下上圖的圖表，可以看出 人類文本 的機率分佈是上下跳動的，而beam search總是傾向選擇高可能性的輸出，因此導致出現這個結果。   

進一步看，這個機率估算是從Open AI 的GPT-2而來，這個模型試圖學會所有人的説法方式。   
對於某人特定的説話方式，會在龐大資料下被generalize。   
對於一些有個性的表達方式，預估出現的機率會變低，即使那個表達也算常見，但在龐大的文本量下呢？    
按照這個思路，也就解釋爲什麽在GPT下，人類文本機率估計會如此跳脫。   

治本的方法是對language model有進一步的約束，除了預測下一個字是什麽，同時也要知道是在什麽背景下説的，讓模型可以求出這個條件概率 - P(background|sentence)   
這個方法還有待時間探討，暫且不表~   

治標來看，可以先試試看sampling的想法~    
既然人類文本出現的機率是上下跳動，有高有底    
機率沒有很高也有可能出現在decode的結果中！   
爲了讓這件事情出現，可以用抽樣的想法。    
每次從候選結果中抽取一個出來，而這個字出現的機率就是抽到它的機率。    

想法很簡單，操作空間也很大，撿起每一個部分來看：   

我們可以控制softmax之後的，候選詞分佈的結果，讓其可以像是poisson distribution或者uniform distribution。    
抽樣輸出的時候，poisson distribution會比較像greed的輸出，而uniform distribution會更加偏向隨機輸出的結果。   

![](https://raw.githubusercontent.com/voidful/voidful_blog/master/assets/post_src/pn_tccontd/img7)


我們每次也不一定需要從所有結果從抽樣，我們可以只拿機率較高的抽樣，免得抽到太無厘頭的字使得整個結果變爛。   

最簡單的想法是抽Top-k個結果出來，比如k=10的話，就是抽頭10個最好的結果出來。   
但有可能第8個結果出來的時候，機率已經十分低了，top-k不會根據當前機率分佈去抽取足夠高的某幾項   

可能馬上想到設置threshold的方法，但我們之前用temperature去控制輸出的機率分佈，threshold也要考慮這個新的機率分佈去設置，我們不希望參數之間是相依的。   

The Curious Case of Neural Text Degeneration 提出了 Nucleus sampling的方法去解決這個問題，想法主要是取 某個纍計機率下的所有項。舉例來看：   
Nucleus sampling 設 0.8   
第一個結果的機率 = 0.66   
第二個結果的機率 = 0.12   
第三個結果的機率 = 0.10   
第四個結果的機率 = 0.06   
第五個結果的機率 = 0.06   
計算其連相加機率   
第一個結果 = 0.66   
第二個結果 = 0.66+0.12 = 0.78   
第三個結果 = 0.66+0.12+0.10 = 0.88   
第四個結果 = 0.66+0.12+0.10+0.06 = 0.94   
第五個結果 = 0.66+0.12+0.10+0.06+0.06 = 1   
由於 0.78 < 0.8 且 0.88 > 0.8   
就會選第一，第二個作爲我們抽樣的樣本。   

以上的 Temperature, top-k / Nucleus sampling 整合起來，也就得到這個讓文本更加靈動的解碼方法。   
huggingface團隊做了pytorch版的code出來   
Code   
[https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317)   
```python
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# Here is how to use this function for top-p sampling
temperature = 1.0
top_k = 0
top_p = 0.9

# Get logits with a forward pass in our model (input is pre-defined)
logits = model(input)

# Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
logits = logits[0, -1, :] / temperature
filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

# Sample from the filtered distribution
probabilities = F.softmax(filtered_logits, dim=-1)
next_token = torch.multinomial(probabilities, 1)
```

他們團隊還用這個方法做了一個聊天機器人,有興趣可以去瞭解下：   

![](https://cdn-images-1.medium.com/max/1200/1*Fn0bcNyyEyqpGq-nCPyoYw.gif)    


[Medium](https://medium.com/@Thomwolf/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313)   

[Github](https://github.com/huggingface/transfer-learning-conv-ai)   





