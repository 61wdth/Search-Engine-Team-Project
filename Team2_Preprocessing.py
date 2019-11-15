import pandas as pd

dataset = pd.read_csv('dataset/lyrics_test.csv', engine='python')
feature = dataset['lyrics']
target = dataset['genre']

for i in range(len(feature)):
    feature[i] = feature[i].lower()                     # 대문자를 소문자 변환
    feature[i] = feature[i].replace("(", "")
    feature[i] = feature[i].replace(")", "")            # '('와 ')' 삭제
    feature[i] = feature[i].replace("[", "")
    feature[i] = feature[i].replace("]", "")            # '['와 ']' 삭제
    feature[i] = feature[i].replace(":", "")            # ':' 삭제
    feature[i] = feature[i].replace("verse", "")        # verse 단어 삭제
    feature[i] = feature[i].replace("1", "")            # 숫자 1 삭제
    feature[i] = feature[i].replace(",", "")            # ',' 삭제
    feature[i] = feature[i].replace(".", "")            # '.' 삭제

dataset['lyrics'] = feature                             # 변환된 가사로 저장
dataset.to_csv(r'NewData4_test.csv')
