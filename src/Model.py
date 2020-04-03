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
            result = "{}  {}  {: .2E}  {}  {}\r".format(tweet_id, predicted_language, score, language, label)
            print(result)
            trace.write(result)
        trace.close()
        print("{:>30s}".format("...Completed Testing..."))
    def evaluate(self):
        print(f"\n●●● Evaluating Model V={self.v} n={self.n} d={self.delta} ●●●")
        # Create evaluation file
        evaluation_file = open(f"eval_{self.v}_{self.n}_{self.delta}.txt", "w", encoding="utf-8")
        # Open trace file to evaluate
        trace_file = open(f"trace_{self.v}_{self.n}_{self.delta}.txt", "r", encoding="utf-8")

        metrics = dict()
        metrics['correct'] = 0
        metrics['total'] = 0
        metrics['Acc'] = 0
        metrics['Precision'] = []
        metrics['Recall'] = []
        metrics['F1'] = []

        for predicted_language in self.language_probabilities.keys():
            metrics[predicted_language] = dict()
            metrics[predicted_language]['TP'] = 0
            metrics[predicted_language]['FP'] = 0
            metrics[predicted_language]['FN'] = 0       # False negative means it did not predict correctly
            metrics['correct'] = 0  # Keeps track of correct for accuracy measure
            metrics['total'] = 0    # Keeps track correct+wrong for accuracy measure

        for line in trace_file.readlines():
            split = line.replace("\n", "").split("  ")
            # Format of trace after split
            # [twitter_id, predicted_lang, score, actual_lang, 'correct/wrong']

            predicted_language = split[1]
            actual_language = split[3]
            if split[4] == 'correct':
                metrics['correct'] = metrics.get('correct') + 1
                metrics['total'] = metrics.get('total') + 1

                # The predicted_language is a True Positive
                metrics[predicted_language]['TP'] = metrics[predicted_language].get('TP') + 1
                # We don't need to keep track of True Negatives, but it would be that
                # all other languages (except predicted language) increment in True Negative

            else:   # label = wrong
                metrics['total'] = metrics.get('total') + 1

                # The predicted language is identifying a False Positive
                metrics[predicted_language]['FP'] = metrics[predicted_language].get('FP') + 1

                # The actual language has a False Negative
                metrics[actual_language]['FN'] = metrics[actual_language].get('FN') + 1

        # Calculate accuracy
        correct = metrics.get('correct')
        total = metrics.get('total')
        accuracy = correct/total
        metrics['Acc'] = accuracy

        F1_total = 0  # Will be used to calculate the macro-F1
        F1_weighted_total = 0  # Will be used to calculate weighted average F1

        for lang in self.language_probabilities.keys():
            language_metric = metrics.get(lang)
            TP = language_metric.get('TP')
            FP = language_metric.get('FP')
            FN = language_metric.get('FN')

            # Calculate precision TP/(TP+FP)
            NaN = "NaN"
            if TP+FP > 0:
                P = TP/(TP+FP)
                metrics['Precision'].append(P)
            else:
                metrics['Precision'].append(np.nan)

            # Calculate recall TP/(TP+FN)
            if TP+FN > 0:
                R = TP/(TP+FN)
                metrics['Recall'].append(R)
            else:
                metrics['Recall'].append(np.nan)

            # Calculate F1 measure (B=1, precision and recall have same importance)
            # F = (B^2 + 1)PR/(B^2P+R)
            B = 1
            if (pow(B, 2)*P+R) > 0:
                F1 = (pow(B, 2) + 1)*P*R/(pow(B, 2)*P+R)
                metrics['F1'].append(F1)
                F1_total += F1
                F1_weighted_total += F1 * self.language_probabilities[lang]
            else:
                metrics['F1'].append(np.nan)

        evaluation_file.write(f"{metrics['Acc']:.4f}\r")

        for val in metrics['Precision']:
            evaluation_file.write(f"{val:.4f}  ")
        evaluation_file.write("\r")

        for val in metrics['Recall']:
            evaluation_file.write(f"{val:.4f}  ")
        evaluation_file.write("\r")

        for val in metrics['F1']:
            evaluation_file.write(f"{val:.4f}  ")
        evaluation_file.write("\r")

        macro_F1 = F1_total/6
        weighted_F1 = F1_weighted_total/6

        evaluation_file.write(f"{macro_F1:>.4f}  {weighted_F1:>.4f}\r")
        print(f"{'Accuracy':15s}{metrics['Acc']:.4f}")
        print(f"{'Languages':15s}{self.language_probabilities.keys()}")
        print(f"{'Precision':15s}{metrics['Precision']}")
        print(f"{'Recall':15s}{metrics['Recall']}")
        print(f"{'F1':15s}{metrics['F1']}")
        print(f"{'Macro-F1':15s}{macro_F1:>.4f}")
        print(f"{'Weighted-F1':15s}{weighted_F1:>.4f}")


