---
title: 細探WordEmbedding
categories:
 - NLP
tags: wordvec
---

nlp_notes
Notes about nlp

NLP常會用到的做法：

基於規則
基於統計
傳統機器學習
神經網絡
基於規則方法，運用構詞原理和句法規則匹配
好處：

有比較好的識別率
不需要大量sample 壞處
難以解決歧意問題
對新內容支持有限
用法：
字典匹配
內容抽取：

匹配算法：
正向最大匹配 / 反向最大匹配
Trie Tree / AC 自動機 / 後綴樹
基於統計的方法，是利用統計方式提取特徵
常見的算法有

最大熵 - 表示信息混亂程度（排除混亂所需要的次數
TF/IDF
TF 在本文檔中出現的頻率
IDF log(總文檔數/出現文檔數)
馬爾科夫 離當前狀態越遠的因素對當前狀態對影響越小
CRF HEMM
random forest
kNN 圖像壓縮
SVM/Logistic Regression 分類
Ada-boost
借鑑生物神經網絡
用數學模型做的人工神經網絡

模拟人的思考
層層遞進
講問題抽象成一個個層次
每一個層次只負責一個特徵

受限伯爾茲曼機 RBM
gibbs sampleing 隨機抽樣方式

根據輸入提取特徵
可以說是一種壓縮
輸入的層數比輸出多
可以壓縮資訊，提取一些特徵
反過來
則可以發現特徵中的細節

Feed-Forward Neural Networks
前向反饋神經網絡

BP误差反向传播算法来更新和训练参数

RNN 在 此上加入有向環
LSTM是RNN的一種
選擇保存訊息和永久儲存