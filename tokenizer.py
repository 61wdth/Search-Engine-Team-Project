def tokenizer(text):
    words = text.split(' ')
    # tokenized_words = []
    str = ""
    for word in words:
        str += word_tokenizer(word) + " "
    return str


def word_tokenizer(word):
    if ('.' not in word) and ('\'' not in word):
        return word
    else:
        if '\'' in word:
            # TODO - 따옴표로 시작해서 따옴표로 끝나는 단어의 따옴표 삭제, 단어 도중에 따옴표가 나오는 경우 따옴표 포함 뒤의 글자 모두 삭제
            if word[0] == '\'' and word[-1] == '\'':
                word = word[1:-1]
        if '\'' in word:
            idx = word.find('\'')
            word = word[:idx]

        if '.' in word:
            # TODO - ".com"으로 끝나는 단어는 토큰화되지 않도록
            if '.com' in word:
                return word
                pass
            # TODO - 마침표로 연결된 단어에서 마침표 앞, 뒤 및 사이에 있는 글자가 모두 1개일 경우 마침표 삭제, 0개 혹은 2개 이상이면 토큰화되지 않도록
            else:
                # 먼저 '.' 나오는 리스트를 만든다.
                dotList = [i for i in range(len(word)) if word.startswith('.', i)]
                # if 0개 혹은 2개 continue
                if (dotList[0] == 1) and (dotList[-1] == len(word) - 2) :
                    check = True
                    for i in range(len(dotList) - 1):
                        if (dotList[i+1] - dotList[i] != 2) :
                            check = False
                            return word
                            pass
                    if(check): word = word.replace('.', '')
                else :
                   return word
        return word


if __name__ == '__main__':
    text = '''i've 'hello' 'hello'world' imlab's PH.D I.B.M snu.ac.kr 127.0.0.1 galago.gif ieee.803.99 naver.com gigabyte.tw pass..fail'''
    print(tokenizer(text))
    #
    # # 추가적으로 검증하기 위해 다음과 같은 Dataset을 tokenize한다.
    # text2 = """ 'i.g.g.g.g.f.f'.g.g.g.g' 'i.g.g.g.g.f.f.'g.g.g.g' .com.1.1 ''''' 'hi''hello' """
    # print("\n",tokenizer(text2))