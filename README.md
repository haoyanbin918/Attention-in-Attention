# Attention in Attention: Modeling Context Correlation for Efficient Video Classification (IEEE TCVST 2022)
This is an official implementaion of paper "Attention in Attention: Modeling Context Correlation for Efficient Video Classification", which has been accepted by IEEE TCVST 2022. [`Paper link`](https://arxiv.org/pdf/2204.09303.pdf)
<div align="center">
  <img src="demo/AIA.pdf" width="700px"/>
</div>


## Updates
### Apr 20, 2022
* Release this V1 version (the version used in paper) to public. Complete codes and models will be released soon.

## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Code](#code)
- [Pretrained Models](#pretrained-models)
  * [Something-Something](#something-something)
    + [Something-Something-V1](#something-something-v1)
    + [Something-Something-V2](#something-something-v2)
  * [Diving48](#Diving48)
  * [EGTEA Gaze+](#EGTEA-Gaze)
- [Train](#Train)
- [Test](#Test)
- [Contibutors](#Contributors)
- [Citing](#Citing)
- [Acknowledgement](#Acknowledgement)

## Prerequisites

The code is built with following libraries:
* PyTorch >= 1.7, torchvision
* tensorboardx

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

## Data Preparation



## Code


## Pretrained Models

Here we provide some of the pretrained models. 


### Something-Something


#### Something-Something-V1

| Model             | Frame * view * clip    | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| AIA(TSN) ResNet50   | 8 * 1 * 1  | 48.5%      | 77.2%     | [link]() |

#### Something-Something-V2

| Model             | Frame * view * clip    | Top-1 Acc. | Top-5 Acc. | Checkpoint |
| ----------------- | ----------- | ---------- | ----------- | ---------------- |
| AIA(TSN) ResNet50   | 8 * 1 * 1  | 60.3%      | 86.4%     | [link]() |

### Diving48
| Model             | Frame * view * clip    | Top-1 Acc. |  Checkpoint |
| ----------------- | ----------- | ---------- | ----------- |
| AIA(TSN) ResNet50   | 8 * 1 * 1  | 79.3%     | [link]() |
| AIA(TSM) ResNet50   | 8 * 1 * 1  | 79.4%     | [link]() |



### EGTEA Gaze
| Model             | Frame * view * clip    | Split1 |  Split2 | Split3 |
| ----------------- | ----------- | ---------- | ----------- | ----------- |
| AIA(TSN) ResNet50   | 8 * 1 * 1  | 63.7%     | 62.1%    | 61.5%  |
| AIA(TSN) ResNet50   | 8 * 1 * 1  | 64.7%     | 63.3%    | 62.2%  |


## Train 

 ```
  ```

## Test 

```
```

## Contributors
GC codes are jointly written and owned by [Dr. Yanbin Hao](https://haoyanbin918.github.io/) and [Dr. Shuo Wang].

## Citing
```bash

```

## Acknowledgement
Thanks for the following Github projects:
- https://github.com/yjxiong/temporal-segment-networks
- https://github.com/mit-han-lab/temporal-shift-module


