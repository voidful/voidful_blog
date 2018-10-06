---
title: Aws Cloud Notes
description: About how to get use of aws service
categories: 
 - 開發筆記
tags: dev AWS Notes
---

# EC2
EC2 is for Virtual Server Hosting   
about pricing :  ttps://aws.amazon.com/tw/ec2/pricing/  
1. download pem key  
2. using ssh connect    
    ```
        ssh -i pem_location ec2-user@public_ip 
    ```
    
Notice:
    
    ```
    對於 Amazon Linux，用戶名稱是 ec2-user。  
    對於 RHEL5，用戶名稱是 root 或 ec2-user。  
    對於 Ubuntu，用戶名稱是 ubuntu。  
    對於 Fedora，用戶名稱是 fedora 或 ec2-user。  
    對於 SUSE Linux，用戶名稱是 root 或 ec2-user。  
    另外，如果 ec2-user 和 root 無法使用，請與您的 AMI 供應商覈實。  
    ```
#  SetUp
For aws linux , use yum to install dependence   
```
 node : sudo yum install nodejs npm --enablerepo=epel  
 ```

#  Run On background
```
screen
```
List all runnung :
```
screen -ls
```
Resume one of the following:
```
screen -r id -d 
```

# aws cli
Refer to : https://github.com/aws/amazon-ecs-cli

# Install  
Download Mac :
```
sudo curl -o /usr/local/bin/ecs-cli https://s3.amazonaws.com/amazon-ecs-cli/ecs-cli-darwin-amd64-latest
```
Download Linux : 
```
sudo curl -o /usr/local/bin/ecs-cli https://s3.amazonaws.com/amazon-ecs-cli/ecs-cli-linux-amd64-latest
```
load cli service :
```
sudo chmod +x /usr/local/bin/ecs-cli
ecs-cli --version
```

#  ecs-cli Configure
```
ecs-cli configure 
--region   
--access-key  
--secret-key 
--cluster (what ever u want in string)

Get access and secret key :
http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html

Region:
us-east-1   US East (N. Virginia)
us-east-2   US East (Ohio)
us-west-1   US West (N. California)
us-west-2   US West (Oregon)
eu-west-1   EU (Ireland)
eu-central-1   EU (Frankfurt)
ap-northeast-1   Asia Pacific (Tokyo)
ap-northeast-2   Asia Pacific (Seoul)
ap-southeast-1   Asia Pacific (Singapore)
ap-southeast-2   Asia Pacific (Sydney)
ap-south-1   Asia Pacific (Mumbai)
sa-east-1   South America (São Paulo)
```

# Start
```
ecs-cli up 
--keypair (name of pem)
--capability-iam 
--size 1 
--instance-type t2.micro
```


