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

test = "abc!DeFg*"
test2 = "a$bcdefgab*cd*e"

create_n_grams(2, test, en_dict, v1_regex)
print(en_dict)
