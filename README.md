# online_retail购物篮商品关联分析
# 项目需求分析
## 1.1 功能需求
关联分析：通过关联规则分析来挖掘商品之间的关联性，主要利用Apriori算法进行频繁项集挖掘和关联规则分析。
商品推荐：基于关联规则的购物篮推荐，通过用户的购物篮信息，推荐与购物篮中商品关联度高的其他商品。
机器学习：利用机器学习方法推荐商品，通过特征提取和相似性计算，提供个性化的推荐服务。
## 1.2 性能需求
高效处理能力：系统需要快速地对大量的用户数据和商品数据进行分析和处理，及时地为用户提供个性化的商品推荐。
稳定性和可靠性：系统需要长时间稳定运行，不会出现频繁的故障和崩溃，确保用户能够持续地获得推荐服务。
## 1.3 安全需求
数据安全保护：确保用户数据和推送数据的安全性，防止数据泄露、篡改和丢失。
用户隐私保护：遵循相关的隐私保护法律法规，尊重用户的隐私权，不非法获取和使用用户的个人信息。
项目分析与设计
## 2.1 技术路线选择
关联分析技术
Apriori算法：选择Apriori算法作为关联分析的核心算法，通过连接步骤和剪枝步骤，逐步生成频繁项集，并根据频繁项集生成关联规则。
改进策略：
设置合理的最小支持度阈值，减少频繁项集的生成数量。
引入并行计算技术，提高算法的处理速度。
优化数据结构，减少数据的存储和访问时间。
机器学习技术
特征提取与相似性计算：
采用TF-IDF算法对商品描述进行特征提取，将商品描述转换为特征向量。
通过计算新商品描述与现有商品描述之间的余弦相似度，找到与新商品最相似的其他商品。
模型选择与训练：
选择合适的机器学习模型，如逻辑回归、支持向量机(SVM)、长短时记忆网络(LSTM)、门控循环单元(GRU)、卷积神经网络(CNN)等。
对商品描述进行分词、去停用词、向量化等预处理，构建模型，训练模型并进行调优。
推荐算法优化
营销效应率最高：以获得最高的营销响应率为目标，优化推荐算法。
销售额最大化：以最大化总体销售额为目标，优化推荐算法。
未消费人群推荐：针对用户未产生消费的情况，优化推荐算法。
## 2.2 系统架构设计
数据层设计
数据存储：采用关系型数据库和非关系型数据库相结合的方式存储数据。
数据处理：对存储的数据进行清洗、转换、整合等处理，以满足分析和推荐的需求。
逻辑层设计
关联分析模块：负责对商品数据进行关联分析，挖掘商品之间的关联规则。
机器学习模块：负责对商品描述进行特征提取和相似性计算，构建机器学习模型，并进行商品推荐。
## 2.3 项目流程设计
数据收集与预处理
收集用户数据：通过电商平台的注册、登录、浏览、购买等环节，收集用户的个人信息、行为数据等。
收集商品数据：从电商平台的商品数据库中获取商品的基本信息、描述、图片、价格、库存等数据。
数据预处理：对收集到的数据进行清洗、转换、整合等处理，以提高数据的质量和可用性。
关联分析与机器学习
关联分析：使用Apriori算法对商品数据进行关联分析，挖掘商品之间的关联规则。
机器学习：对商品描述进行特征提取和相似性计算，构建机器学习模型，并进行商品推荐。
推荐结果生成与优化
推荐结果生成：根据关联分析和机器学习的结果，生成个性化的商品推荐。
项目实现与测试
## 3.1 项目实现
数据预处理实现
使用Python的pandas库读取数据并处理数据，使用matplotlib库、seaborn库进行数据的可视化。
关联规则分析实现
使用mlxtend库中的apriori函数和association_rules函数进行关联规则分析。
机器学习模型实现
使用sklearn库中的TfidfVectorizer和cosine_similarity函数进行特征提取和相似性计算。
## 3.2 项目测试
集成测试
将项目中的各个模块和功能进行集成测试，确保它们能够协同工作，实现整个项目的功能。
单元测试
对项目中的各个模块和函数进行单元测试，确保其功能正确性和稳定性。
性能测试
对项目进行性能测试，评估其处理速度、响应时间、稳定性等性能指标。
项目成果与应用价值
## 4.1 项目成果
系统开发成果
成功开发了一个完整的个性化商品推荐系统，实现了关联分析、机器学习和推荐算法优化等功能。
技术应用成果
在关联分析方面，成功应用了Apriori算法进行频繁项集挖掘和关联规则分析。
在机器学习方面，利用TF-IDF算法和余弦相似度计算，实现了商品描述的特征提取和相似性计算，构建了有效的机器学习模型。
数据处理成果
对大量的用户数据和商品数据进行了有效的收集、存储和处理，提高了数据的质量和可用性。

# 基于协同化过滤的商品推荐系统
## 问题建模
本项目的数据集给出了15万左右的用户以及12万左右的商品, 以及对应的经过脱敏处理的用户特征和经过预处理的商品特征，旨在为用户推荐50个其可能购买的商品。
## 推荐系统架构方案
  本项目采用传统的召回＋排序的方案。在召回模块采用deepwalk， node2vec，item_feature, itemCF四种方法进行多路召回，为每位用户召回1000个商品。在排序阶段采用wide&deep模型，对召回的1000个商品进行排序。将排序所得的分数依据商品点击量进行后处理，来增大对非热门商品的曝光度。最后根据处理后的分数为每位用户推荐50个商品。

## 文件结构
数据来源于阿里天池平台开源数据。

    │  feature_list.csv                               # List the features we used in ranking process
    │  project_structure.txt                          # The tree structure of this project
    ├─ build_graph_model.py                          # Build deepwalk model and node2vec model
    ├─ final_rank.py                          # Build wide&deep network
    ├─ final_solution.py                          # Main program
    ├─ recall_function.py                          # Functions used to recall items
    ├─ item_feat.pkl                          # Item feature after PCA
    ├─ top100_recall_feature.pkl                          # Recalled 100 items for each user by using item_feature
    ├─ top300_recall_deepwalk_result.pkl                          # Recalled 300 items for each user by using deepwalk
    ├─ top300_recall_node2vec_result.pkl                          # Recalled 300 items for each user by using node2vec
    ├─ topk_recall.pkl                          # Recalled 1000 items for each user by combining all ways
    ├─ train_eval_rank.pkl                          # Cross validation set after ranking
    ├─ wide_and_deep.h5                          # Wide&Deep model using full training set
    ├─ wide_and_deep_no_cv.h5                          # Wide&Deep model using training set except cross validation set
    ├─ data                                           # Origin dataset
    │  ├─ underexpose_test
    │  └─ underexpose_train
    ├─ readme.md
    ├─ deepwalk_offline.bin                                      # deepwalk model
    └─ node2vec_offline.bin                                      # node2vec model

## Python库环境依赖

    tensorflow==2.3.1
    scikit-learn==0.23.2
    joblib==0.17.0
    networkx==2.1
    gensim==3.8.3
    pandas==0.25.1
    numpy==1.18.5
    tqdm==4.26.0
