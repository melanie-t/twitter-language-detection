from src.NaivesBayesClassification import train_model, calculate_score


class Model:
    def __init__(self, v, n, delta, training_path):
        self.v = v
        self.n = n
        self.delta = delta
        self.language_probabilities, self.ngram_probabilities = train_model(self.v, self.n, self.delta, training_path)

    def predict_language(self, tweet):
        language_scores = dict()
        for language in self.language_probabilities.keys():
            score = calculate_score(tweet, self.v, self.n, self.language_probabilities.get(language),
                                    self.ngram_probabilities.get(language))
            language_scores[language] = score
        predicted = max(zip(language_scores.values(), language_scores.keys()))
        return predicted

    def test(self, test_path):
        # Open test file
        test_set = open(test_path, "r", encoding="utf-8")
        # Create trace file
        trace = open("trace_{}_{}_{}.txt".format(self.v, self.n, self.delta), "w", encoding="utf-8")
        print("{:>27s}".format("...Begin Testing..."))
        for line in test_set.readlines():
            split = line.replace("\n", "").split("\t")
            tweet_id = split[0]
            language = split[2]
            tweet = split[3]
            score, predicted_language = self.predict_language(tweet)
            if predicted_language == language:
                label = "correct"
            else:
                label = "wrong"
            # Scientific Notation
            # Source: https://kite.com/python/answers/how-to-print-a-number-in-scientific-notation-in-python
            result = "{}  {}  {:.2E}  {}  {}\r".format(tweet_id, predicted_language, score, language, label)
            print(result)
            trace.write(result)
        trace.close()
        print("{:>30s}".format("...Completed Testing..."))


