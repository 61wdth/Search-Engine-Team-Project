from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
# TODO - Import appropriate modules
from nltk.util import ngrams
import math
from sklearn import linear_model

"""3-1"""
with open("bible.txt", 'r') as f:
    text = f.read()
s_list = ['.', ',', '?', '!', ';', ':', '\'s', '(', ')', '“', '”', '’', '=', '>', '+', '<', '&', '#', '·', '&', '←']
for c in s_list:
    text = text.replace(c, '')
# 소문자로 변환
text = text.lower()

# TODO - Make zipf_law function
def zipf_law(tokenized_words):
    total_word_count = len(tokenized_words)
    word_count_dict = {}
    # TODO - 단어를 key로, 단어의 등장 빈도를 value로 갖는 dictionary 생성: word_cout_dict[word] = freq
    for token in tokenized_words:
        if token in word_count_dict:
            word_count_dict[token] += 1
        else:
            word_count_dict[token] = 1
    word_rank_list = []
    # TODO - word_count_dict를 freq 높은 순으로 나열하여 순서대로 word_rank_list에 추가
    # index가 rank에 대응(index = rank - 1) : word_rank_list[rank-1] = word
    for key, value in sorted(word_count_dict.items(), key=lambda item: item[1], reverse=True):
        word_rank_list.append(key)
    col_head = "Word".ljust(20) + " Freq".ljust(11) + " r".ljust(7) + "Pr".ljust(20) + "r*Pr"

    freq_list = []
    prob_list = []
    rank_list = []

    for i in range(0, len(word_rank_list)):
        # TODO - word, freq에 알맞은 값 지정
        word = word_rank_list[i]
        freq = word_count_dict[word]
        ocur_prob = freq / total_word_count
        # freq가 같은 경우 rank도 같아야 하지만 현재 word_rank_list의 index는 1씩 증가하며 모두 다른 값을 가지므로
        # 바로 rank = i + 1을 적용할 수 없음
        # TODO - 조건문을 적용하여 같은 freq값을 갖는 단어가 여러 개일 경우 rank 처리(아래 if False에서 False 대신 조건문 적용)
        if i == 0:
            rank = 1
        elif freq == freq_list[-1]:
            rank = rank_list[-1]
        else:
            rank = i + 1
        freq_list.append(freq)
        prob_list.append(ocur_prob)
        rank_list.append(rank)
        # if using Unigram
        if ' ' not in word:
            if i==0:
                print(col_head)
            print(word.ljust(20), str(freq).ljust(10), str(rank).ljust(5), '%.17f' % ocur_prob, '%.6f' % (rank * ocur_prob))
        # if using Bi, Tri-gram
        # else:
        #   print(col_head)
        #   print(word, str(freq), str(rank), '%.17f' % ocur_prob, '%.6f' % (rank * ocur_prob))

    return freq_list, prob_list, rank_list

# TODO - zipf's law plot (Represent X,Y axis name!)
tokenized_bible = word_tokenize(text)           # 다른 tokenizer 사용 가능
ufreq_list, uprob_list, urank_list = zipf_law(tokenized_bible)
ulog_prob_list = []
ulog_rank_list = []
for f in uprob_list:
    ulog_prob_list.append(math.log(f))
for r in urank_list:
    ulog_rank_list.append(math.log(r))

x1 = np.array(ulog_rank_list)
y1 = np.array(ulog_prob_list)
x1 = x1.reshape(len(x1), 1)
y1 = y1.reshape(len(y1), 1)
regr = linear_model.LinearRegression()
regr.fit(x1, y1)
pred_y1 = regr.predict(x1)

x = np.arange(0,12,1)
y = (-1)*x + math.log(0.1)
plt.plot(x,y, color = 'black', label = 'x+y = k')
plt.scatter(x1, y1, color = 'blue', label = 'UnigramData')
plt.plot(x1, pred_y1, color = 'blue', label = 'UnigramLine')
plt.xlim(0,12)
plt.ylim(-15,0)
plt.title('Zipf\'s Law', fontsize=15)
plt.xlabel('Log(Rank)')
plt.ylabel('Log(Probability)')
plt.legend()
plt.grid(True)
plt.show()



"""3-2"""
# TODO - Bigram & Trigram Using NLTK library methods

bi_tokens = ngrams(tokenized_bible, 2)
tri_tokens = ngrams(tokenized_bible, 3)


bigrams_list = []
trigrams_list = []
for token in bi_tokens:
    bigrams_list.append((token[0] + ' ' + token[1]))
for token in tri_tokens:
    trigrams_list.append((token[0] + ' ' + token[1] + ' ' + token[2]))

bfreq_list, bprob_list, brank_list = zipf_law(bigrams_list)
tfreq_list, tprob_list, trank_list = zipf_law(trigrams_list)

# TODO - zipf's law plot(better to plot multiple lines(Uni, Bi, Tri) in one plot to compare)
# (Represent X,Y axis name!)
blog_prob_list = []
blog_rank_list = []
tlog_prob_list = []
tlog_rank_list = []
for bf in bprob_list:
    blog_prob_list.append(math.log(bf))
for br in brank_list:
    blog_rank_list.append(math.log(br))
for tf in tprob_list:
    tlog_prob_list.append(math.log(tf))
for tr in trank_list:
    tlog_rank_list.append(math.log(tr))

x1 = x1.reshape(len(x1))
y1 = y1.reshape(len(y1))
x2 = np.array(blog_rank_list)
x3 = np.array(tlog_rank_list)
y2 = np.array(blog_prob_list)
y3 = np.array(tlog_prob_list)

#아래는 선형 회귀분석에 대한 부분입니다.
print('\n'+ '\033[1m' + '--Result of Linear Regression(Uni/Bi/Tri)--' + '\033[0m')
print('correlation1:', np.corrcoef(x1,y1)[0][1])
print('correlation2:', np.corrcoef(x2,y2)[0][1])
print('correlation3:', np.corrcoef(x3,y3)[0][1])
x1 = x1.reshape(len(x1), 1)
y1 = y1.reshape(len(y1), 1)
x2 = x2.reshape(len(x2), 1)
y2 = y2.reshape(len(y2), 1)
x3 = x3.reshape(len(x3), 1)
y3 = y3.reshape(len(y3), 1)
regr2 = linear_model.LinearRegression()
regr2.fit(x2, y2)
regr3 = linear_model.LinearRegression()
regr3.fit(x3, y3)
pred_y2 = regr2.predict(x2)
pred_y3 = regr3.predict(x3)

print('Unigram coefficient:', regr.coef_, 'Unigram intercept:', regr.intercept_)
print('Bigram coefficient: ', regr2.coef_, 'Bigram intercept:', regr2.intercept_)
print('Trigram coefficient: ', regr3.coef_, 'Trigram intercept:', regr3.intercept_)

plt.scatter(x1, y1, color = 'blue', label = 'Unigram')
plt.scatter(x2, y2, color = 'green',label = 'Bigram')
plt.scatter(x3, y3, color = 'red', label = 'Trigram')
plt.plot(x1, pred_y1, color = 'blue', label = 'UnigramLine')
plt.plot(x2,pred_y2, c='green', label='BigramLine')
plt.plot(x3,pred_y3, c='red', label='TrigramLine')
plt.xlim(0,12)
plt.ylim(-15,0)
plt.title("Uni vs. Bi vs. Tri" , fontsize=15)
plt.xlabel('Log(Rank)')
plt.ylabel('Log(Probability)')
plt.grid(True)
plt.legend()
plt.show()

#빈도가 n인 단어의 비율 1/n(n+1) 확인
uwordset_size=len(set(tokenized_bible))
bwordset_size=len(set(bigrams_list))
twordset_size=len(set(trigrams_list))

ufreq_rate_list=[]
bfreq_rate_list=[]
tfreq_rate_list=[]
for ufreq in sorted(list(set(ufreq_list)))[:30]:
    ufreq_rate_list.append(ufreq_list.count(ufreq) / uwordset_size)
for bfreq in sorted(list(set(bfreq_list)))[:30]:
    bfreq_rate_list.append(bfreq_list.count(bfreq) / bwordset_size)
for tfreq in sorted(list(set(tfreq_list)))[:30]:
    tfreq_rate_list.append(tfreq_list.count(tfreq) / twordset_size)

zipf_rate_list=[]
for freq in range(1, 31):
    zipf_rate_list.append(1/(freq*(freq+1)))

#freq-rate 비교 log 씌워서
log_zipf_rate=[]
for rate in zipf_rate_list:
    log_zipf_rate.append(math.log(rate))

log_ufreq_list=[]
for freq in sorted(list(set(ufreq_list)))[:30]:
    log_ufreq_list.append(math.log(freq))
log_ufreq_rate_list=[]
for rate in ufreq_rate_list:
    log_ufreq_rate_list.append(math.log(rate))
log_bfreq_list=[]
for freq in sorted(list(set(bfreq_list)))[:30]:
    log_bfreq_list.append(math.log(freq))
log_bfreq_rate_list=[]
for rate in bfreq_rate_list:
    log_bfreq_rate_list.append(math.log(rate))
log_tfreq_list=[]
for freq in sorted(list(set(tfreq_list)))[:30]:
    log_tfreq_list.append(math.log(freq))
log_tfreq_rate_list=[]
for rate in tfreq_rate_list:
    log_tfreq_rate_list.append(math.log(rate))


plt.plot(np.array(log_ufreq_list), np.array(log_zipf_rate), c='black', label='Zipf\'s Law')
plt.plot(np.array(log_ufreq_list), np.array(log_ufreq_rate_list), c='blue', marker='.', ls=':', label='Unigram')
plt.xlim(0,3.5)
plt.ylim(-9, 0)
plt.title('Unigram: Freq-Rate')
plt.xlabel('Log(Frequency)')
plt.ylabel('Log(Rate)')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(np.array(log_bfreq_list), np.array(log_zipf_rate), c='black', label='Zipf\'s Law')
plt.plot(np.array(log_bfreq_list), np.array(log_bfreq_rate_list), c='green', marker='.', ls=':', label='Bigram')
plt.xlim(0,3.5)
plt.ylim(-9, 0)
plt.title('Bigram: Freq-Rate')
plt.xlabel('Log(Frequency)')
plt.ylabel('Log(Rate)')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(np.array(log_tfreq_list), np.array(log_zipf_rate), c='black', label='Zipf\'s Law')
plt.plot(np.array(log_tfreq_list), np.array(log_tfreq_rate_list), c='red', marker='.', ls=':', label='Trigram')
plt.xlim(0,3.5)
plt.ylim(-9, 0)
plt.title('Trigram: Freq-Rate')
plt.xlabel('Log(Frequency)')
plt.ylabel('Log(Rate)')
plt.legend()
plt.grid(True)
plt.show()