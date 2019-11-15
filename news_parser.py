from bs4 import BeautifulSoup
from nltk.tag import pos_tag
from urllib.request import urlopen
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
# Install WordCloud
# Using This : pip install
# TODO - Download WordNet for Lemmatizer
# TODO - Using This : nltk.download(wordnet)
from wordcloud import WordCloud, ImageColorGenerator
from os import path, getcwd
import numpy as np
from PIL import Image

# 1-1. print title, author, submission date, abstract content, subjects using BeautifulSoup
# TODO - fill in your team's url
url = 'https://arxiv.org/abs/1711.00937'

data = urlopen(url).read()
doc = BeautifulSoup(data, 'html.parser')

# TODO - print title
title = doc.find('meta', attrs={'name': 'citation_title'}).get('content')
print("Title: ", title)

# TODO - print authors
author = ""
for a in doc.find_all('meta', attrs={'name': 'citation_author'}):
    author += a.get("content") + ". "
print("Author: ", author)

# TODO - print Submission date
date = doc.find('meta', attrs={'name': 'citation_date'}).get('content')
print("Date: ", date)

# TODO - print abstract
abstract = doc.find('meta', attrs={'property': 'og:description'}).get('content')
print("Abstract: ", abstract)

# TODO - print subjects
subjects = doc.find('span', attrs={'class': 'primary-subject'}).get_text()
print("subjects: ", subjects)


# 1-2. Tokenize abstract content by words and POS-Tag tokenized words
# TODO - tokenize article content
tokenized_words = word_tokenize(abstract)
print('Tokenized Words List\n',tokenized_words)


# 1-3. sort tokenized words by frequency and plot WordCloud
# TODO - POS_Tag tokenized words
tagged_list = pos_tag(tokenized_words)
print('Tokenized Words with POS List\n',tagged_list)

token_count = {}
for token in tokenized_words:
    token_count[token] = tokenized_words.count(token)
ordered_tokens = sorted(token_count.items(), key=lambda item: item[1], reverse = True)
print(ordered_tokens)

# 1-4. plot WordCloud and apply stopwords to WordCloud
# using 'token_count' dictionary to plot wordcloud
wordcloud = WordCloud().generate_from_frequencies(token_count)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # TODO - plot new wordcloud which represent abstract more effectively
# Defining WordNetLemmatizer
wl = WordNetLemmatizer()

# Define j for ordering after deleting un-meaningful tokens
j=0

# Lemmatize Noun by deleting 's'
# Use Lemmatizer for Verb
for i in range(len(tokenized_words)):
    if(tagged_list[i][1] in ['NNS', 'NNP']) :
        if(tagged_list[i][0][-1] == 's') :
            tokenized_words[i-j] = tokenized_words[i-j][:-1]
    if tagged_list[i][1] in ['VB', 'VBZ', 'VBN', 'VBP'] :
        tokenized_words[i-j] = wl.lemmatize(tokenized_words[i-j],'v')
    if tagged_list[i][1] in ['.',',',')','(',':','``',"''",'IN','DT','CC','TO',':','WDT','RB','WRB','MD','PRP','PRP$','CD']:
        del(tokenized_words[i-j])
        j = j+1

# Uppercase tokenized_word for counting tokens
tokenized_words = [x.upper() for x in tokenized_words]

# Delete BE verb which is not meaningful
for i in range(tokenized_words.count('BE')):
    tokenized_words.remove('BE')
print(tokenized_words)

# Count token by its frequency after processing
word_count = {}
for word in tokenized_words:
   word_count[word] = tokenized_words.count(word)
word_count[subjects[:subjects.find(' (')]] = int(max(word_count.values())*1.5)
print(word_count)

# wordcloud.png for mask
d = getcwd()
mask = np.array(Image.open(path.join(d, "wordcloud.png")))

# Generating Improved WordCloud
wc = WordCloud(width = 1024, height = 1024,background_color = "black", mask = mask).fit_words(word_count)
image_colors = ImageColorGenerator(mask)
plt.imshow(wc.recolor(color_func = image_colors), interpolation = "bilinear")
plt.axis("off")
plt.show()
