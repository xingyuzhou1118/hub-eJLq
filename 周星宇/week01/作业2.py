import pandas as pd
import os
import jieba
from pyarrow.dataset import dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from tensorboard.compat.tensorflow_stub.tensor_shape import vector
from openai import OpenAI
# 车载电台的输入文本意图识别
#先导入数据集
dataset = pd.read_csv("dataset.csv",sep="\t",names=['text','label'],nrows=100)
#对原始数据集进行处理
input_sentence = dataset['text'].apply(lambda x: " ".join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(input_sentence.values)#统计词表，生成词向量
input_feature = vector.transform(input_sentence.values)#将输入文本转换为词向量
#训练模型
model = KNeighborsClassifier()
model.fit(input_feature,dataset['label'].values)
# print(dataset['label'].value_counts())

#使用大模型对文本进行分类
client = OpenAI(
    api_key = "sk-91f5996ed4xxxxx5ab636f1a2",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",# 大模型厂商的地址，阿里云

)

def text_classify_using_ml(text:str) -> str:
    """
    使用机器学习算法对文本进行分类
    :param text: 输入的文本
    :return: 文本分类结果
    """
    #使用训练好的模型对输入文本进行分类
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])

    # 使用模型预测文本分类结果，返回值是数组，[0]表示取第一个元素（预测的类别标签）
    return model.predict(test_feature)[0]

def text_classify_using_llm(text:str) -> str:
    """
    使用大语言模型对文本进行分类
    :param text: 输入的文本
    :return: 文本分类结果
    """
    completion = client.chat.completions.create(
        model="qwen3-max-preview",
        messages=[
            {"role": "system", "content": "你是一个分本分类助手，你需要根据用户输入的文本，判断其所属的分类"},
            {"role": "user", "content": f"""帮我进行文本分类，请将以下文本进行分类：{text};
            分类的类别只能从以下选项中选择：
            FilmTele-Play
            Video-Play
            Music-Play
            Radio-Listen
            Alarm-Update
            Weather-Query
            Travel-Query
            HomeAppliance-Control
            Calendar-Query
            TVProgram-Play
            Audio-Play
            Other
            """},

        ]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    """
    # 使用pandas读取数据集
    data = pd.read_csv('dataset.csv',sep='\t',names=['text','label'],nrows=None)
    print(data.head(10))
    # 数据集的样本维度
    print("数据集的样本维度：",data.shape)
    # 数据集的标签分布
    print("数据集的标签分布：",data["label"].value_counts())
    """
    test = "帮我播放音乐"
    # print(text_classify_using_ml(test))
    print("使用ML模型分类结果：",text_classify_using_ml(test))
    print("使用LLM模型分类结果：",text_classify_using_llm(test))
