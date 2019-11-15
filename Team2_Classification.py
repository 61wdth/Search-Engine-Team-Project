import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

dataset = pd.read_csv('NewData4.csv', engine='python')
feature = dataset['lyrics']
target = dataset['genre']

dataset_test = pd.read_csv('NewData4_test.csv', engine='python')
feature_test = dataset_test['lyrics']
target_test = dataset_test['genre']

lemmatizer = WordNetLemmatizer() #lemmatizer 미리 생성
pt = PorterStemmer() #porterstemmer 미리 생성
# TODO - Choose 'test_size' and 'random_state'

X_train, _, Y_train, _ = train_test_split(feature, target, test_size=0.15, random_state=0) #test_size = 0.15

			# TODO - Make your own features
			# Example : TF-IDF vector
			#X_train_feature = TfidfTransformer().fit_transform(CountVectorizer(stop_words = 'english', max_features=20).fit_transform(X_train)).toarray()
			#X_test_feature = TfidfTransformer().fit_transform(CountVectorizer(stop_words = 'english', max_features=20).fit_transform(X_test)).toarray()
			#tokenizer 함수 조절해볼것 

			# TODO - 2-1-1. Build pipeline for Naive Bayes Classifier


#밑의 4줄은 Navie Bayes Classifier pipeline 형성 위해 미리 인스턴스 생성해 놓는 것.
cv_nb = CountVectorizer(ngram_range = (1, 1),stop_words = 'english',  max_features = 152, decode_error = 'replace', tokenizer = word_tokenize, preprocessor = pt.stem) 
titrans_nb = TfidfTransformer(sublinear_tf = True)
ss_nb = StandardScaler(with_mean = False)
mn_nb = MultinomialNB(alpha = 10)
#pipeline 형성 및 fitting
clf_nb = Pipeline([('vect', cv_nb), ('tfidf', titrans_nb), ('scaler', ss_nb), ('clf', mn_nb)])
clf_nb.fit(X_train, Y_train)


# TODO - 2-1-2. Build pipeline for SVM Classifier

#밑의 5줄은 SVM pipeline 형성 위해 미리 인스턴스 생성해 놓는 것.
cv_svm = CountVectorizer(ngram_range = (1, 2), stop_words = 'english',  max_features=147, decode_error = 'replace', tokenizer = word_tokenize, preprocessor = pt.stem)
titrans_svm = TfidfTransformer(sublinear_tf = True)
ss_svm = StandardScaler(with_mean = False)
sel_svm = SelectKBest(f_classif, k=147)
lisvc_svm = SVC(C = 1.2, kernel = 'rbf', class_weight = 'balanced', gamma = 'scale')
#pipeline 형성 및 fitting
clf_svm = Pipeline([('vect', cv_svm), ('tfidf', titrans_svm), ('scaler', ss_svm), ('selector', sel_svm), ('clf', lisvc_svm)])
clf_svm.fit(X_train, Y_train)

#predict
predicted_nb = clf_nb.predict(feature_test)
predicted_svm = clf_svm.predict(feature_test)

#결과 출력
# print("accuracy_nb : %d / %d" % (np.sum(predicted_nb==Y_test), len(Y_test)))
# print(np.sum(predicted_nb==Y_test)/len(Y_test))

print("accuracy_svm : %d / %d" % (np.sum(predicted_svm==target_test), len(target_test)))
print(np.sum(predicted_svm==target_test)/len(target_test))


# with open('model_nb.pkl', 'wb') as f1:
#     pickle.dump(clf_nb, f1)

# '''
# '''
# with open('model_svm.pkl', 'wb') as f2:
#     pickle.dump(clf_svm, f2)

