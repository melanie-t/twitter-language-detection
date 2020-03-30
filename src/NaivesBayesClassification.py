import re


def build_model(vocab, n, lang, tweet, vocabulary, all_lang, ngram_count):
    # print(lang, tweet)
    for i in range(n - 1, len(tweet)):
        start_index = i - (n - 1)
        ngram = tweet[start_index:i + 1]
        if vocab == 0:
            tweet = tweet.lower()
        if valid_ngram(vocab, ngram):
            # Check if ngram hasn't been added to vocabulary yet
            if ngram not in vocabulary:
                # Add ngram to vocabulary and initialize the value in all other languages
                # print("New ngram", ngram)
                vocabulary.append(ngram)
                for key in all_lang.keys():
                    all_lang[key][ngram] = 0
            # Update ngram count in given lang
            ngram_count[lang] = ngram_count.get(lang) + 1
            all_lang[lang][ngram] = all_lang[lang].get(ngram) + 1
            # print("\t Updated ", ngram, all_lang[lang][ngram])


def valid_ngram(vocab, ngram):
    lang_regex = ''
    if vocab == 0:
        return bool(re.match("^[a-z]+$", ngram))        # V=0 Lowercase and only letters
    elif vocab == 1:
        return bool(re.match("^[a-zA-Z]+$", ngram))     # V=1 Distinguish upper and lower case
    elif vocab == 2:
        return ngram.isalpha()                          # V=2 isalpha
    else:
        return False


def train_model(vocab, n, smoothing):
    all_languages = dict()
    ngram_count = dict()
    language_count = dict()
    vocabulary = []

    # Initialize dictionaries
    languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
    for language in languages:
        all_languages[language] = dict()
        ngram_count[language] = 0
        language_count[language] = 0

    tweet = "abc!DeFg*"
    tweet2 = "a$bcdefgab*cd*e"
    f = open("OriginalDataSet/training-1.txt", "r", encoding="utf-8")
    training_set = f.readlines()
    for line in training_set:
        split = line.replace('\n', '').split("\t")
        lang = split[2]
        tweet = split[3]

        language_count[lang] = language_count.get(lang) + 1
        build_model(vocab, n, lang, tweet, vocabulary, all_languages, ngram_count)

    for lang in all_languages.keys():
        print(lang, language_count[lang], ngram_count[lang], all_languages[lang])


def build_vocab0(n):
    vocabulary = dict()
    a = 97
    if n == 1:
        for i in range(0, 26):
            vocabulary[chr(a+i)] = 0
    elif n == 2:
        for i in range(0, 26):
            for j in range(0, 26):
                vocabulary[chr(a+i)+chr(a+j)] = 0
    elif n == 3:
        for i in range(0, 26):
            for j in range(0, 26):
                for k in range(0, 26):
                    vocabulary[chr(a+i)+chr(a+j)+chr(a+k)] = 0
    return vocabulary


def build_vocab1_2(n, isalpha):
    vocabulary = dict()
    A = 65
    a = 97

    if n == 1:
        for i in range(0, 26):
            vocabulary[chr(A+i)] = 0
            vocabulary[chr(a+i)] = 0

    elif n == 2:
        for i in range(0, 26):
            for j in range(0, 26):
                vocabulary[chr(A+i)+chr(A+j)] = 0        # 1: All upper-case
                vocabulary[chr(a+i)+chr(a+j)] = 0        # 2: All lower-case
                if isalpha:
                    vocabulary[chr(A+i)+chr(a+j)] = 0        # 3: First char upper, second char lower
                    vocabulary[chr(a+i)+chr(A+j)] = 0        # 4: First char lower, second char upper

    elif n == 3:
        for i in range(0, 26):
            for j in range(0, 26):
                for k in range(0, 26):
                    vocabulary[chr(A+i)+chr(A+j)+chr(A+k)] = 0       # 1: All upper case (UUU)
                    vocabulary[chr(a+i)+chr(a+j)+chr(a+k)] = 0       # 5: All lower case (LLL)
                    if isalpha:
                        vocabulary[chr(a+i)+chr(a+j)+chr(A+k)] = 0       # 6: Lower, Lower, Upper (LLU)
                        vocabulary[chr(A+i)+chr(A+j)+chr(a+k)] = 0       # 2: Upper, Upper, Lower (UUL)
                        vocabulary[chr(A+i)+chr(a+j)+chr(A+k)] = 0       # 3: Upper, Lower, Upper (ULU)
                        vocabulary[chr(a+i)+chr(A+j)+chr(A+k)] = 0       # 4: Lower, Upper, Upper (LUU)
                        vocabulary[chr(a+i)+chr(A+j)+chr(a+k)] = 0       # 7: Lower, Upper, Lower (LUL)
                        vocabulary[chr(A+i)+chr(a+j)+chr(a+k)] = 0       # 8: Upper, Lower, Lower (ULL)

    return vocabulary


# train_model(0, 1, 1)
# vocab0(3)
vocab = build_vocab1_2(1, False)
print(vocab)
print(len(vocab))


