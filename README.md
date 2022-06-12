# Keyphrase Extraction Algorithm

无监督学习：中文关键词抽取（Keyphrase Extraction），基于LDA与PageRank（TextRank， TPR， Salience Rank， Single TPR）

## Introduction

|   Algorithm   |                        Intro                        |                             ref                              |
| :-----------: | :-------------------------------------------------: | :----------------------------------------------------------: |
|   TextRank    |           将PageRank应用于文本关键词抽取            |        [paper](https://aclanthology.org/W04-3252.pdf)        |
|      TPR      | 首次将主题（Topic）信息整合到 PageRank 计算的公式中 |        [paper](https://aclanthology.org/D10-1036.pdf)        |
|  Single TPR   |            单词迭代计算的Topic  PageRank            | [paper](https://biblio.ugent.be/publication/5974208/file/5974209.pdf) |
| Salience Rank |             引入显著性的Topic  PageRank             |         [paper](https://aclanthology.org/P17-2084/)          |

## Dependencies
  - sklearn
  - jieba  0.42.1
  - networkx 2.5
  - numpy 1.20.1
  - pandas 1.2.4
  - matplotlib 3.3.4
  - queue 0.6.3

## Data

本项目采用新浪新闻8个领域（体育，娱乐，彩票，房产，教育，游戏，科技，股票）的新闻数据共800条作为实验数据。

数据集位于`data/data.xlsx`下，由两列组成，第一列content存放新闻标题和新闻的正文内容，第二列是type是该新闻的话题类型。

在模型训练过程只需要利用excel文件中的content列，第二列是根据提取的关键词来衡量提取的准确性。

### 如何使用自己的数据

按照`data.xlsx`的数据格式放置你的数据，只需要content列即可。

## Config

`config`目录下可以配置：

- `jieba`分词库的自定义词典`jieba_user_dict.txt`
- 添加停用词（stopwords）`stop_words.txt`

## Usage
```
# TextRank
python main.py --alg text_rank
# TPR
python main.py --alg tpr
# Single TPR
python main.py --alg single_tpr
# Salience Rank
python main.py
```

## Result

- TextRank前十条数据提取关键词结果

```
0  :  训练;大雨;球员;队员;队伍;雨水;热身赛;事情;球队;全队;国奥;影响;情况;比赛;伤病
1  :  战术;姑娘;首战;比赛;过程;记者;主帅;交锋;信心;剪辑;将士;软肋;世界杯;夫杯;遭遇
2  :  冠军;活动;女士;文静;游戏;抽奖;俱乐部;眼镜;大奖;特等奖;奖品;现场;环节;教练;球队
3  :  俱乐部;球员;工资;危机;宏运;球队;奖金;管理;老队员;教练;笑里藏刀;前提;集体;集团;经验
4  :  警方;立案侦查;总局;产业;电话;足球;外界;消息;公安部门;依法;中体;主席;裁判;检察机关;委员会
5  :  比赛;鹿队;机会;命中率;队员;联赛;调整;开赛;压力;包袱;外援;主场;状态;体育讯;金隅
6  :  火箭;球队;比赛;原因;时间;效率;开局;事实;教练组;变化;轨道;过程;漫长;判断能力;时机
7  :  胜利;球队;队友;火箭;篮板;比赛;关键;垫底;句式;小牛;新浪;战绩;体育讯;活塞;时间
8  :  火箭;交易;活塞;球队;球员;情况;筹码;价值;命运;市场;续约;掘金;遭遇;球星;核心
9  :  湖人;比赛;球队;后卫;揭幕战;沙农;时间;出场;阵容;板凳;火力;外线;念头;贡献;证明
10  :  公牛;球员;球队;教练;数据;比赛;能力;体系;主教练;命中率;交流;研究;水平;记者;小时
```

## Reference

  - Text Rank: Mihalcea and Tarau. 2004. Textrank: Bringing order into texts.
  - TPR: Liu et al. 2010. Automatic keyphrase extraction via topic decomposition.
  - Single TPR: Sterckx et al. 2015. Topical word importance for fast keyphrase extraction.
  - Salience Rank: Nedelina et al . 2017.Salience Rank: Efficient Keyphrase Extraction with Topic Modeling.
