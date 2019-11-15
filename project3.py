import surprise
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import numpy as np
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.evaluate import GridSearch
from surprise.prediction_algorithms.matrix_factorization import NMF
import pandas as pd
from surprise import dataset
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from sklearn.model_selection import train_test_split
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.slope_one import SlopeOne
from surprise.prediction_algorithms.random_pred import NormalPredictor
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise.prediction_algorithms.knns import KNNBaseline
'''
def get_top_n(algo, testset, id_list, n=10, user_based=True):
    results = defaultdict(list)
    if user_based:
        # TODO - testset의 데이터 중 user id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장(2점)
        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for dataset in testset:
            if dataset[0] in id_list:
                testset_id.append(dataset[0])
        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            # TODO - results는 user_id를 key로, [(item_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary(2점)
            if uid not in results.keys():
                results[uid] = [(iid, est)]
            else:
                results[uid].append((iid, est))
    else:
        # TODO - testset의 데이터 중 item id가 id_list 안에 있는 데이터만 따로 testset_id라는 list로 저장(2점)
        # Hint: testset은 (user_id, item_id, default_rating)의 tuple을 요소로 갖는 list
        testset_id = []
        for dataset in testset:
            if dataset[1] in id_list:
                testset_id.append(dataset[1])
        predictions = algo.test(testset_id)
        for uid, iid, true_r, est, _ in predictions:
            # TODO - results는 item_id를 key로, [(user_id, estimated_rating)의 tuple이 모인 list]를 value로 갖는 dictionary(2점)
            if iid not in results.keys():
                results[iid] = [(uid, est)]
            else:
                results[iid].append((uid, est))

    for id, ratings in results.items():
        # TODO - rating 순서대로 정렬하고 top-n개만 유지(2점)
        ordered_ratings = sorted(ratings, reverse=True, key=lambda rating: rating[1])
        results[id] = ordered_ratings[:n]

    return results
'''

file_path = 'data/user_game_log.dat'
reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
'''
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
'''
'''
# 1 - User-based Recommendation
uid_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# TODO - 1-1. KNNBasic, cosine
sim_cos = {'name': 'cosine'}
algo1_1 = surprise.KNNBasic(sim_options=sim_cos)
algo1_1.fit(trainset)
results = get_top_n(algo1_1, testset, uid_list, n=10, user_based=True)
with open('1-1_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 1-2. KNNWithMeans, pearson
sim_pearson = {'name': 'pearson'}
algo1_2 = surprise.KNNWithMeans(sim_options=sim_pearson)
algo1_2.fit(trainset)
results = get_top_n(algo1_2, testset, uid_list, n=10, user_based=True)
with open('1-2_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# 2 - Item-based Recommendation
iid_list = ['Dota 2', 'Football Manager 2012','FIFA Manager 11', 'NBA 2K12', 'Call of Duty United Offensive'
            'Blood Bowl Chaos Edition', 'Team Fortress 2', 'FINAL FANTASY XIII', 'Counter-Strike', 'SimCity 4 Deluxe']
# TODO - 2-1. KNNBasic, cosine
algo = None
algo.fit(trainset)
results = get_top_n(algo, testset, iid_list, n=10, user_based=False)
with open('2-1_results.txt', 'w') as f:
    for iid, ratings in sorted(results.items(), key=lambda x: (x[0])):
        f.write('Item ID %s top-10 results\n' % iid)
        for uid, score in ratings:
            f.write('User ID %s\tscore %s\n' % (uid, str(score)))
        f.write('\n')

# TODO - 2-2. KNNWithMeans, pearson
algo = None
algo.fit(trainset)
results = get_top_n(algo, testset, iid_list, n=10, user_based=False)
with open('2-2_results.txt', 'w') as f:
    for iid, ratings in sorted(results.items(), key=lambda x: (x[0])):
        f.write('Item ID %s top-10 results\n' % iid)
        for uid, score in ratings:
            f.write('User ID %s\tscore %s\n' % (uid, str(score)))
        f.write('\n')

# 3 - Matrix-factorization Recommendation
# TODO - 3-1. SVD, n_factors=100, n_epochs=50, biased=False
algo = None
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('3-1_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 3-2. SVD, n_factors=200, n_epochs=100, biased=True
algo = None
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('3-2_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 3-3. SVD++, n_factors=100, n_epochs=50
algo = None
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('3-3_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')

# TODO - 3-4. SVD++, n_factors=50, n_epochs=100
algo = None
algo.fit(trainset)
results = get_top_n(algo, testset, uid_list, n=10, user_based=True)
with open('3-4_results.txt', 'w') as f:
    for uid, ratings in sorted(results.items(), key=lambda x: int(x[0])):
        f.write('User ID %s top-10 results\n' % uid)
        for iid, score in ratings:
            f.write('Item ID %s\tscore %s\n' % (iid, str(score)))
        f.write('\n')
'''
# TODO - 4. Make your own Best Model(Don't need to save result file(txt))
best_algo_mf = None
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
#print(testset)
#algo4 = SlopeOne()
#cross_validate(algo4, data, measures= ['RMSE'], cv=3, verbose=True)

#algo4 = NMF(n_factors = 93)
'''
train_set, test_set = train_test_split(data, test_size = 0.25)
predictions = algo4.fit(train_set).test(test_set)
accuracy.rmse(predictions)
'''
'''
df = pd.DataFrame(data.raw_ratings, columns = ["user","game","rate","id"])
df = df.drop(["id"],axis = 1)
#df = df.pivot(index='user', columns='game', values='rate')
print(df.head(20))
#df = df.fillna(0)
#Newdata = Dataset.load_from_df(df, reader)

df = df["rate"].fillna(0)
Newdata = Dataset.load_from_df(df, reader)

predictions = algo4.fit(Newdata).test(data)
accuracy.rmse(predictions)
'''
#param_grid = {'method': 'als' ,'n_factors': [10,20,30,40], 'biased': [True, False]}
#g = GridSearch(BaselineOnly, param_grid = param_grid, measures = ['RMSE'])
#g.evaluate(data)
#print(g.best_params)
#pring(g.best_score['RMSE'])
'''
minValue = 2.0
minij = [0,0]
for i in range(1,30):
    for j in range(1,20):
        bsl_options = {'method': 'als', 'n_factors': 93,'n_epochs': 23, 'reg_i': i, 'reg_u': j}
        algo4 = BaselineOnly(bsl_options)
        print([i,j])
        nowValue = float(cross_validate(algo4, data, measures= ['RMSE'], cv=3)['test_rmse'].mean())
        if minValue > nowValue:
            minValue = nowValue
            minij = [i,j]
print(minValue)       
print(minij)
[3,7]
'''
'''
algo4 = NormalPredictor()
print(float(cross_validate(algo4, data, measures = ['RMSE'], cv = 3)['test_rmse'].mean()))
'''
'''
algo4 = NMF(n_factors = 5)
predictions = algo4.fit(trainset).test(testset)
accuracy.rmse(predictions)
'''
#bsl_options = {'method': 'als', 'n_factors': 93,'n_epochs':23, 'reg_i': 3, 'reg_u':7}
#algo4 = BaselineOnly(bsl_options)
#predictions = algo4.fit(trainset).test(testset)
#accuracy.rmse(predictions)
algo4 = KNNBaseline(k = 90, sim_options = {'name': 'pearson_baseline', 'shrinkage': 0})
cross_validate(algo4,data,measures = ['RMSE'], cv = 3, verbose = True)