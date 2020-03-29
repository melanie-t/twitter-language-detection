import re


def build_model(vocab, n, lang, tweet, all_lang, ngram_count):
    for i in range(n - 1, len(tweet)):
        start_index = i - (n - 1)
        n_gram = tweet[start_index:i + 1]
        if validate_ngram(vocab, n_gram):
            # Update counts for ngram
            ngram_count[lang] = ngram_count.get(lang) + 1
            count = all_lang[lang].get(n_gram)
            if count is None:
                all_lang[lang][n_gram] = 1
                # print("Added", n_gram, "to dict")
            else:
                all_lang[lang][n_gram] = count + 1
                # print("Updated", n_gram, "count", language_dict[n_gram])


def validate_ngram(vocab, ngram):
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
        build_model(vocab, n, lang, tweet, all_languages, ngram_count)

    for lang in all_languages.keys():
        print(lang, language_count[lang], ngram_count[lang], all_languages[lang])


train_model(0, 1, 1)
