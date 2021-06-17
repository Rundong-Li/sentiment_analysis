import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm

stop_words = "  0123456789,?!！.．,:;@，。？：；—…&@#~:;、……～＆＠+*-\t\xad\u3000\u2003\xa0\ufeff＃“”‘’〝〞 \"'＂＇´＇'()（）【】《》＜＞﹝﹞" \
             "<>()[]«»‹›［］「」｛｝〖〗『』"
Path = './dataset/online_shopping_10_cats.csv'

# 读取文件
print('[Preprocess]: reading data.')
df = pd.read_csv(Path)
df = df.drop(['cat'], axis=1)
df_new = df.copy(deep=True)

# 去停用词，分词
print('[Preprocess]: cutting words.')
n, _ = df.shape
for i in tqdm(range(n)):
    label, review = df.iloc[i]
    review = str(review)
    for word in stop_words:
        review = review.replace(word, '')
    if review == '':  # 扔掉空行
        df_new = df_new.drop(labels=[i], axis=0)
        continue
    review = " ".join(jieba.lcut(review))
    df_new.loc[i] = [label, review]
print('[Preprocess]: we have {} examples.'.format(df_new.shape[0]))

# 划分训练验证集测试集
df_new = df_new.reindex(np.random.permutation(df_new.index))  # 打乱顺序
df_train = df_new[0:50000]
df_valid = df_new[50000:]
df_test = df_new[50000:]
print('[Preprocess]: train set contains {} examples.'.format(df_train.shape[0]))
print('[Preprocess]: valid set contains {} examples.'.format(df_valid.shape[0]))
print('[Preprocess]: test set contains {} examples.'.format(df_test.shape[0]))

# 保存数据
print("[Preprocess]: writing files.")
df_train.to_csv('./Raw_Data/train.txt', index=False, sep=" ", header=False)
df_valid.to_csv('./Raw_Data/validation.txt', index=False, sep=" ", header=False)
df_test.to_csv('./Raw_Data/test.txt', index=False, sep=" ", header=False)
print("[Preprocess]: data preprocess finished.")
