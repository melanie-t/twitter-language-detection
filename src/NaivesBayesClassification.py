import re


def build_model(vocab, n, tweet, language_dict):
    for i in range(n - 1, len(tweet)):
        start_index = i - (n - 1)
        n_gram = tweet[start_index:i + 1]
        if validate_ngram(vocab, n_gram):
            count = language_dict.get(n_gram)
            if count is None:
                language_dict[n_gram] = 1
                print("Added", n_gram, "to dict")
            else:
                language_dict[n_gram] = count + 1
                print("Updated", n_gram, "count", language_dict[n_gram])


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
    all_lang = dict()
    eu_dict = dict()
    ca_dict = dict()
    gl_dict = dict()
    es_dict = dict()
    en_dict = dict()
    pt_dict = dict()

    all_lang['eu'] = eu_dict
    all_lang['ca'] = ca_dict
    all_lang['gl'] = gl_dict
    all_lang['es'] = es_dict
    all_lang['en'] = en_dict
    all_lang['pt'] = pt_dict

    tweet = "abc!DeFg*"
    tweet2 = "a$bcdefgab*cd*e"
    f = open("OriginalDataSet/training-1.txt", "r")
    training_set = f.readlines()
    for line in training_set:
        split = line.split("\t")
        print(split)
        lang = split[2]
        tweet = split[3]

        build_model(vocab, n, tweet, all_lang[lang])

    print(all_lang)


train_model(0, 3, 1)
