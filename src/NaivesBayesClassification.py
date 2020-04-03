import re
import unicodedata
from math import log


def build_model(v, n, lang, tweet, ngram_frequency, ngram_total):
    # print(lang, tweet)
    for i in range(n-1, len(tweet)):
        start_index = i - (n - 1)
        ngram = tweet[start_index:i + 1]
        # ngram = tweet[i] + "|" + tweet[start_index:i]
        if v == 0:
            tweet = tweet.lower()

        if valid_ngram(v, ngram):
            character = ngram[0]
            # print(ngram)
            # For v=2, we only add character occurences and ngrams to vocab as we see them in training set
            if v == 2:
                count = ngram_frequency[lang].get(ngram)
                if count is None:   # Initialize new ngram
                    ngram_frequency[lang][ngram] = 1
                    ngram_total[lang] = 1
                else:
                    ngram_frequency[lang][ngram] = count + 1     # Update existing
                    ngram_total[lang] = ngram_total.get(lang) + 1
            # For v=0, v=1, we have already initialized each language dictionary
            else:
                # character_count[lang][character] = character_count[lang].get(character) + 1
                ngram_frequency[lang][ngram] = ngram_frequency[lang].get(ngram) + 1
                ngram_total[lang] = ngram_total.get(lang) + 1


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


def train_model(v, n, delta, training_path):
    ngram_frequency = dict()  # Contains the frequency count of each ngram for each language
    ngram_total = dict()    # Contains the number of ngrams in each language model
    tweet_count = dict()    # Contains the number of tweets for each language
    vocabulary = dict()     # The vocabulary of the type specified (0,1,2)

    # Build vocabulary
    if v == 0:
        vocabulary = build_vocab0(n)
    elif v == 1:
        vocabulary = build_vocab1(n)
    # When v=2, we don't initialize the vocabulary

    # Initialize dictionaries
    languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
    for language in languages:
        ngram_frequency[language] = vocabulary.copy()
        ngram_total[language] = 0
        tweet_count[language] = 0
        tweet_count['total'] = 0

    # We load the training data and process each line and pass the values into build_model function
    # which will break down the tweet into ngrams and insert into the appropriate language model
    f = open(training_path, "r", encoding="utf-8")
    training_set = f.readlines()
    # print("{:>27s}".format("...Begin Training..."))
    for line in training_set:
        split = line.replace("\n", "").split("\t")
        lang = split[2]
        tweet = split[3]

        tweet_count[lang] = tweet_count.get(lang) + 1
        tweet_count['total'] = tweet_count['total'] + 1
        build_model(v, n, lang, tweet, ngram_frequency, ngram_total)

    smooth(n, delta, ngram_frequency, ngram_total)
    print("   {:>30s} {:>30s} {:>30s}".format("tweet_count", "ngram_total", "vocab_size"))
    for lang in ngram_frequency.keys():
        print(lang, "{:>30.2f} {:>30.2f} {:>30.2f}".format(tweet_count[lang], ngram_total[lang], len(ngram_frequency[lang])))
    print("total {:>27d}".format(tweet_count['total']))

    language_probabilities = calculate_language_probabilities(ngram_frequency, tweet_count)
    ngram_probabilities = calculate_ngram_probabilities(ngram_frequency, ngram_total)

    # print('language probabilities', language_probabilities)
    # print('ngram probabilities', ngram_probabilities)

    # print("{:>31s}".format("...Completed Training...\n"))
    return language_probabilities, ngram_probabilities


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
                # vocabulary[chr(a+i)+"|"+chr(a+j)] = 0
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


def smooth(n, delta, ngram_frequency_count, ngram_total):
    for lang in ngram_frequency_count.keys():
        for ngram in ngram_frequency_count[lang].keys():
            ngram_frequency_count[lang][ngram] = ngram_frequency_count[lang].get(ngram) + delta
        vocab_size = len(ngram_frequency_count[lang])
        ngram_total[lang] = ngram_total.get(lang) + delta * vocab_size ** n


# calculate_language_probabilities
# This function calculates the probabilities of each language
def calculate_language_probabilities(ngram_frequency, tweet_count):
    lang_probabilities = dict()
    for lang in ngram_frequency.keys():
        lang_probabilities[lang] = tweet_count[lang]/tweet_count['total']
    return lang_probabilities


# calculate_ngram_probabilities function
# This function calculates the probabilities of each ngram in the language
def calculate_ngram_probabilities(ngram_frequency, ngram_total):
    ngram_probabilities = dict()
    for lang in ngram_frequency.keys():
            total_ngrams = ngram_total.get(lang)
            ngram_probabilities[lang] = dict()
            if ngram_total.get(lang) != 0:
                for ngram in ngram_frequency[lang].keys():
                    ngram_count = ngram_frequency[lang].get(ngram)
                    ngram_probabilities[lang][ngram] = ngram_count/total_ngrams
    return ngram_probabilities


def calculate_score(tweet, v, n, lang_probability, ngram_probability):
    score = log(lang_probability, 10)    # Initialize score by adding probability of language first
    for i in range(n-1, len(tweet)):
        start_index = i - (n - 1)
        ngram = tweet[start_index:i + 1]
        if valid_ngram(v, ngram):
            ngram_score = ngram_probability.get(ngram)
            if ngram_score is not None:
                score += log(ngram_score, 10)
            else:
                return 0
    return score