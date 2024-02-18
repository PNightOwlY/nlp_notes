from .file_util import *

def read_jd_sentiment_cls_data(path):
    # https://modelscope.cn/datasets/DAMO_NLP/jd/files
    # 商品评论情感预测
    df = read_csv(path=path,sep=',')
    dataset = []
    for i in range(len(df)):
        dataset.append(
            {'text': df.iloc[i]['sentence'],
             "label": df.iloc[i]['label']}
        )
    return dataset


def main():
    read_jd_sentiment_cls_data("../../data/text_classification/jd_sentiment_cls/train.csv")
    
if __name__ == "__main__":
    main()