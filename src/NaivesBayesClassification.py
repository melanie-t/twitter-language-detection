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


en_dict = dict()
v0_regex = "^[a-z]+$"          # V=0, lowercase and only letters
v1_regex = "^[a-zA-Z]+$"       # V=1, Distinguish upper and lower case
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

test = "abc!DeFg*"
test2 = "a$bcdefgab*cd*e"

create_n_grams(2, test, en_dict, v1_regex)
print(en_dict)
