import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('NewData4.csv', engine='python')
feature = dataset['lyrics']
target = dataset['genre']

pt = PorterStemmer()

# TODO - Data preprocessing and clustering
# TODO - Set Randomness parameters to specific value(ex: random_state in KMeans etc.) Or Save KMeans model to pickle file
#각 노래의 가사마다 CountVectorizer 함수를 이용해서 vector로 만들고
#그 vector에 대해 TfidfTransformer 함수를 이용해서 tf*idf를 요소로 가지는 새로운 vector 생성
#각각의 vector는 fit_transform 함수로 구한다
data_trans = TfidfTransformer(sublinear_tf=True).fit_transform( \
                              CountVectorizer(ngram_range=(1,1), stop_words="english", max_features=152, \
                                              decode_error='ignore', tokenizer=word_tokenize, \
                                              preprocessor=pt.stem).fit_transform(feature))
sc = StandardScaler(with_mean=False)
sc_data_trans = sc.fit_transform(data_trans)

# # V measure 점수를 가장 높게 만드는 cluster수를 iteration을 통해서 구한다
# # 1부터 30까지의 range에 대해 cluster number를 대입하고 500번 iteration을 하여 각각의 점수를 구한 후
# # 각 cluster number 당 점수의 평균, 표준편차, 최댓값, 최솟값을 출력한다
# # 가장 최댓값이 크게 나온 상위 5개의 cluster number 중 평균이 크고 표준편차가 작은 경우를 최종 cluster number로 지정
result = []
for clusterNo in range(15, 31):
   cnt = 0
   clst_list = []
   while True:
      cnt += 1
      if cnt <= 5:
         clst = KMeans(n_clusters=clusterNo, init='k-means++', max_iter=500, algorithm='full', \
                       random_state=0,precompute_distances='auto')      # 같은 cluster number에 대해 5번 시행한 결과에 대해
         clst.fit(sc_data_trans)                                        # 그 결과에 대한 통계량을 이용하여 cluster number를 결정
         score = metrics.v_measure_score(target, clst.labels_)
         clst_list.append(score)
         print(score)
      else:
         break
   average = np.mean(clst_list)
   stan_deviation = np.std(clst_list)
   maximum = max(clst_list)
   minimum = min(clst_list)
   result.append([clusterNo, average, stan_deviation, maximum, minimum])

for a in result:
    print(a)

clusterNo_final = 15                                                    # clusterNo가 15일 때 점수가 가장 높게 나온다
clst_final = KMeans(n_clusters=clusterNo_final, init='k-means++', \
                    max_iter=500, algorithm='full', random_state=0,\
                    precompute_distances='auto')                        # cluster의 개수를 15으로 지정한 상태에서 데이터 군집화 진행
clst_final.fit(sc_data_trans)
score_final = metrics.v_measure_score(target, clst_final.labels_)
print("clusterNo:", clusterNo_final, score_final)

with open('cluster.pkl', 'wb') as f3:                                   # pickle 파일로 저장
    pickle.dump(clst_final, f3)

clusters = clst_final.labels_.tolist()                                  #.labels.tolist(): 군집화 후 각 데이터의 결과를 array로 표현
labels = target
colors = {0: '#dc143c', 1: '#008000', 2: '#bfff00', 3: '#0080ff', 4: '#8977ad'}
#0: 심홍색 / 1: 초록색 / 2: 라임색 / 3: 바다색 / 4: 밝은 보라

pca = PCA(n_components=2).fit_transform(sc_data_trans.toarray())        # data_trans를 array로 변환 후 PCA 함수를 이용해서 차원 축소(2차원 vector들의 행렬로)
xs, ys = pca[:, 0], pca[:, 1]                                           # 각 vector의 x좌표(xs), y좌표끼리(ys) 새로운 vector 생성
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))                     # dataframe의 label이 cluster(15)인 경우
# df = pd.DataFrame(dict(x=xs, y=ys, label=labels))                       # dataframe의 label이 genre(5)인 경우
print(df)
groups = df.groupby('label')                                            # 같은 label을 갖는 데이터끼리 grouping

#set up plot
fig, ax = plt.subplots(figsize=(17,9))                                  # set size
ax.margins(0.05)                                                        # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#cluster에 대한 df인 경우
for idx, group in groups:
   ax.plot(group.x, group.y, marker='o', linestyle='', ms=8, mec='none')
   ax.set_aspect('auto')
   ax.tick_params(\
      axis='x',                                                         # changes apply to the x-axis \
      which='both',                                                     # both major and minor ticks are affected \
      bottom='off',                                                     # ticks along the bottom edge are off \
      top='off',                                                        # ticks along the top edge are off \
      labelbottom='off')
   ax.tick_params(\
      axis='y',                                                         # changes apply to the y-axis \
      which='both',                                                     # both major and minor ticks are affected \
      left='off',                                                       # ticks along the bottom edge are off \
      top='off',                                                        # ticks along the top edge are off \
      labelleft='off')
plt.show()

#label에 대한 df인 경우
# cnt = 0
# for idx, group in groups:                                               # idx가 genre로 나타나기 때문에 숫자로 변형하기 위해 cnt 활용
#    ax.plot(group.x, group.y, marker='o', linestyle='', color=color_label[cnt], ms=8, mec='none')
#    ax.set_aspect('auto')
#    ax.tick_params(\
#       axis='x',                                                         # changes apply to the x-axis \
#       which='both',                                                     # both major and minor ticks are affected \
#       bottom='off',                                                     # ticks along the bottom edge are off \
#       top='off',                                                        # ticks along the top edge are off \
#       labelbottom='off')
#    ax.tick_params(\
#       axis='y',                                                         # changes apply to the y-axis \
#       which='both',                                                     # both major and minor ticks are affected \
#       left='off',                                                       # ticks along the bottom edge are off \
#       top='off',                                                        # ticks along the top edge are off \
#       labelleft='off')
#    cnt += 1
# plt.show()