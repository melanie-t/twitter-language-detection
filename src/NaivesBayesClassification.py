import re
import unicodedata


def build_model(v, n, lang, tweet, all_lang, ngram_count):
    # print(lang, tweet)
    for i in range(n - 1, len(tweet)):
        start_index = i - (n - 1)
        ngram = tweet[start_index:i + 1]
        if v == 0:
            tweet = tweet.lower()

        if valid_ngram(v, ngram):
            # print(ngram)
            # For v=2, we only add ngrams to vocab as we see them in training set
            if v == 2:
                count = all_lang[lang].get(ngram)
                if count is None:
                    all_lang[lang][ngram] = 1       # Initialize new ngram
                else:
                    all_lang[lang][ngram] = count + 1     # Update existing ngram count
                    ngram_count[lang] = ngram_count.get(lang) + 1

            # For v=0, v=1, we have initialized each language dictionary
            else:
                ngram_count[lang] = ngram_count.get(lang) + 1
                all_lang[lang][ngram] = all_lang[lang].get(ngram) + 1


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


def train_model(v, n, delta):
    all_languages = dict()
    ngram_count = dict()
    tweet_count = dict()
    vocabulary = dict()

    # Build vocabulary
    if v == 0:
        vocabulary = build_vocab0(n)
    elif v == 1:
        vocabulary = build_vocab1(n)
    # When v=2, we don't initialize the model

    # Initialize dictionaries
    languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
    for language in languages:
        all_languages[language] = vocabulary.copy()
        ngram_count[language] = 0
        tweet_count[language] = 0
        tweet_count['total'] = 0

    f = open("OriginalDataSet/training-tweets.txt", "r", encoding="utf-8")
    training_set = f.readlines()
    for line in training_set:
        split = line.replace("\n", "").split("\t")
        lang = split[2]
        tweet = split[3]

        tweet_count[lang] = tweet_count.get(lang) + 1
        tweet_count['total'] = tweet_count['total'] + 1
        build_model(v, n, lang, tweet, all_languages, ngram_count)

    print("   {:>15s} {:>15s} {:>15s}".format("tweet_count", "ngram_count", "vocab_size"))
    for lang in all_languages.keys():
        print(lang, "{:>15d} {:>15d} {:>15d}".format(tweet_count[lang], ngram_count[lang], len(all_languages[lang])))
    print('------------------')
    print("total {:>12d}".format(tweet_count['total']))


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


def build_vocab1(n):
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
                vocabulary[chr(A+i)+chr(a+j)] = 0        # 3: First char upper, second char lower
                vocabulary[chr(a+i)+chr(A+j)] = 0        # 4: First char lower, second char upper

    elif n == 3:
        for i in range(0, 26):
            for j in range(0, 26):
                for k in range(0, 26):
                    vocabulary[chr(A+i)+chr(A+j)+chr(A+k)] = 0       # 1: All upper case (UUU)
                    vocabulary[chr(a+i)+chr(a+j)+chr(a+k)] = 0       # 5: All lower case (LLL)
                    vocabulary[chr(a+i)+chr(a+j)+chr(A+k)] = 0       # 6: Lower, Lower, Upper (LLU)
                    vocabulary[chr(A+i)+chr(A+j)+chr(a+k)] = 0       # 2: Upper, Upper, Lower (UUL)
                    vocabulary[chr(A+i)+chr(a+j)+chr(A+k)] = 0       # 3: Upper, Lower, Upper (ULU)
                    vocabulary[chr(a+i)+chr(A+j)+chr(A+k)] = 0       # 4: Lower, Upper, Upper (LUU)
                    vocabulary[chr(a+i)+chr(A+j)+chr(a+k)] = 0       # 7: Lower, Upper, Lower (LUL)
                    vocabulary[chr(A+i)+chr(a+j)+chr(a+k)] = 0       # 8: Upper, Lower, Lower (ULL)

    return vocabulary