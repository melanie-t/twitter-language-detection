import re


def create_n_grams(n, str, language_dict, language_regex):
    for i in range(n-1, len(str)):
        start_index = i-(n-1)
        n_gram = str[start_index:i+1]
        count = language_dict.get(n_gram)
        # Check if
        if re.match(language_regex, n_gram):
            if count is None:
                language_dict[n_gram] = 1
                print("Added", n_gram, "to dict")
            else:
                language_dict[n_gram] = count+1
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
