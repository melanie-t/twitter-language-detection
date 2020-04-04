import re
from math import log
from src.Vocabulary import build_vocab0, build_vocab1


def update_count(v, n, lang, tweet, ngram_frequency, ngram_total):
    # print(lang, tweet)
    for i in range(n-1, len(tweet)):
        start_index = i - (n - 1)
        ngram = tweet[start_index:i + 1]
        # ngram = tweet[i] + "|" + tweet[start_index:i]

        if valid_ngram(v, ngram):
            # print(ngram)
            # For v=2, we only add character occurences and ngrams to vocab as we see them in training set
            if v != 0 and v != 1:
                count = ngram_frequency[lang].get(ngram)
                if count is None:   # Initialize new ngram
                    for key in ngram_frequency.keys():
                        # Add new ngram to all language dictionaries
                        ngram_frequency[key][ngram] = 0

            ngram_frequency[lang][ngram] = ngram_frequency[lang].get(ngram) + 1     # Update existing
            ngram_total[lang] = ngram_total.get(lang) + 1

            # # For v=0, v=1, we have already initialized each language dictionary
            # else:
            #     # character_count[lang][character] = character_count[lang].get(character) + 1
            #     ngram_frequency[lang][ngram] = ngram_frequency[lang].get(ngram) + 1
            #     ngram_total[lang] = ngram_total.get(lang) + 1


def valid_ngram(vocab, ngram):
    if vocab == 0:
        return bool(re.match("^[a-z]+$", ngram))        # V=0 Lowercase and only letters
    elif vocab == 1:
        return bool(re.match("^[a-zA-Z]+$", ngram))     # V=1 Distinguish upper and lower case
    else:
        return ngram.isalpha()


def train_model(v, n, delta, training_path):
    ngram_frequency = dict()  # Contains the frequency count of each ngram for each language
    ngram_total = dict()    # Contains the number of ngrams in each language model
    tweet_count = dict()    # Contains the number of tweets for each language
    vocabulary = dict()     # The vocabulary of the type specified (0,1,2)

    # Build vocabulary
    if v == 0:
        vocabulary = build_vocab0(n)
    elif v == 1 or v == 3:
        vocabulary = build_vocab1(n)
    # When v=2, we don't initialize the vocabulary

    # Initialize dictionaries
    languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
    for language in languages:
        ngram_frequency[language] = vocabulary.copy()
        ngram_total[language] = 0
        tweet_count[language] = 0
        tweet_count['total'] = 0

    # We load the training data and process each line and pass the values into update_count function
    # which will break down the tweet into ngrams and insert into the appropriate language model
    f = open(training_path, "r", encoding="utf-8")
    training_set = f.readlines()
    # print("{:>27s}".format("...Begin Training..."))
    for line in training_set:
        if line and not line.isspace():
            split = line.split("\t")
            lang = split[2]
            tweet = pre_process_tweet(v, split[3])

            tweet_count[lang] = tweet_count.get(lang) + 1
            tweet_count['total'] = tweet_count['total'] + 1
            update_count(v, n, lang, tweet, ngram_frequency, ngram_total)

    smooth(delta, ngram_frequency, ngram_total)
    print(f"   {'tweet_count':>30s} {'ngram_total':>30s} {'vocab_size':>30s}")
    for lang in ngram_frequency.keys():
        print(f"{lang} {tweet_count[lang]:>30.2f} {ngram_total[lang]:>30.2f} {len(ngram_frequency[lang]):>30.2f}")
    print(f"total {tweet_count['total']:>27d}")

    language_probabilities = calculate_language_probabilities(ngram_frequency, tweet_count)
    ngram_probabilities = calculate_ngram_probabilities(ngram_frequency, ngram_total)

    # print('language probabilities', language_probabilities)
    # print('ngram probabilities', ngram_probabilities)

    # print("{:>31s}".format("...Completed Training...\n"))
    return language_probabilities, ngram_probabilities


def smooth(delta, ngram_frequency_count, ngram_total):
    for lang in ngram_frequency_count.keys():
        for ngram in ngram_frequency_count[lang].keys():
            ngram_frequency_count[lang][ngram] = ngram_frequency_count[lang].get(ngram) + delta
        vocab_size = len(ngram_frequency_count[lang])
        ngram_total[lang] = ngram_total.get(lang) + delta * vocab_size


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
                if ngram_score != 0:
                    score += log(ngram_score, 10)
            else:
                # The test set has an ngram not accounted for in the model, so probability is 0
                return 0

    return score


def pre_process_tweet(v, tweet):
    if v == 0:
        tweet = tweet.lower()
    tweet = tweet.replace("\n", "")
    # Source: https://stackoverflow.com/questions/24399820/expression-to-remove-url-links-from-twitter-tweet/24399874
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\S+", "", tweet)

    return tweet

