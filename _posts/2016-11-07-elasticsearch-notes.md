---
title: ElasticSearch Notes
categories:
 - 開發筆記
tags: ElasticSearch
---

ElasticSearch可以說是一個高效的搜索引擎
同時也可以說是一個數據庫

# ES的特性：
**Base On**
 - 開源 基於lucene

**Store Data**
 - schemea free--不需要定義檔案格式
 - document oriented--基於文檔儲存

**Communicate**
 - RESTful Api
 - 返回 Json 格式

**Scale**
 - 分佈式
 - 多租戶 -- 多用戶環境下保持資料隔離性
 - 良好衝突管理機制
 - 高可用性
 
**Search**
 - 支持full text search 全文搜索
 - 實時搜索/分析

# 安裝和配置
步驟：
 1. 下載 最新版 elasticsearch : https://www.elastic.co/downloads/elasticsearch
 2. 安裝kibana和sense  
    mac 建議用homebrew下載安裝,能省下不少時間
 3. 安裝分詞插件  
    elasticsearch 自帶的分詞插件對中文支持十分不好,需要安裝分詞插件，目前搜索到ik和jieba  
    jieba：https://github.com/huaban/elasticsearch-analysis-jieba
 4. 到這一步基本上安裝完畢，到ES目錄下的./bin,啟動ES  
    之後將會用RESTful api與其通訊  
    
    ```sh
    Action : GET,PUT,DELETE,POST
    +
    URI : Unique Resource
    +
    Content : head +json body
    +
    Status Code : 200,400
    ```
    
 5. 設cluster和mapping  
    ```sh
    PUT--localhost:9200/index
    {
        "settings" : {
            "number_of_shards" : 1,
            "number_of_replicas" : 0
    
        },
        "mappings" : {
            "type" : {
                "_all" : { "enabled" : false },
                "properties" : {
                    "field" : { "type" : "string", "analyzer" : "name" }
                }
            }
        }
    }
    ```
 6. 加入數據
 
     ```sh
     PUT http://localhost:9200 /index/type/id
     {
         "field" : "value"
     }
     ```
     
  7. 檢查 是否加入成功 和 analyzer有否正常工作  
    搜尋所有結果：  
      ```
      GET http://localhost:9200/index/type/_search
      ```
     analyzer分詞測試:  
      ```
      PUT http://localhost:9200/index/document/number
      json:{
          content
      }
      ```

# ElasticSearch於傳統數據庫結構上的區別
Relational DB ： Databases -> Tables -> Rows -> Columns  
Elastic search ： Index   -> Types  -> Documents -> Fields  

Full architecture of ES :  
Cluster ->Node ->Index(Shard|Replication) -> Documents -> Fields



# ElasticSearch-Jieba

Install elasticsearch and install jieba plugin:  
https://github.com/huaban/elasticsearch-analysis-jieba  

After install it set the mapping:  
remember to set the index document and field  analyzer：
  - index 主要用于索引分词，分词粒度较细
  - search 主要用于查询分词，分词粒度较粗
  - other 全角转半角、大写转小写、字符分词
```sh
PUT--localhost:9200/index
{
    "settings" : {
        "number_of_shards" : 1,
        "number_of_replicas" : 0

    },
    "mappings" : {
        "document" : {
            "_all" : { "enabled" : false },
            "properties" : {
                "field" : { "type" : "string", "analyzer" : "jieba_index", "search_analyzer" : "jieba_search" }
            }
        }
    }
}
```

To update the jieba dict
go to elasticsearch/plugins/jieba/dic
  - sougou.dict --courpus from sougou
  - stopwords.txt --stopwords will take out from the text
  - user.dict --the dict can append
after edit it , need to restart elasticsearch
> jieba support Traditional chinese
> formatting syntax is to make it as readable
to upload data used restful api
```sh
PUT http://localhost:9200/index/document/number
json:{
    content
}
```

To test upload success or not, can try to make a query:
```sh
http://localhost:9200/index/document/_search?q=content:value
```

To test jieba  
```sh
post
http://localhost:9200/test/_analyze?analyzer=jieba_search
body:
{text}
```

about how es counting the score:http://www.cnblogs.com/richaaaard/p/5254988.html