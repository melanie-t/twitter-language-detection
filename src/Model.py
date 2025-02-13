from src.NaivesBayesClassification import train_model, calculate_score, pre_process_tweet
import numpy as np


class Model:
    def __init__(self, v, n, delta):
        self.v = v
        self.n = n
        self.delta = delta
        print(f"●●● Initialize Model V={self.v} n={self.n} d={self.delta} ●●●")
        self.language_probabilities = dict()
        self.ngram_probabilities = dict()

    def train(self, training_path):
        print(f"\n●●● Training Model V={self.v} n={self.n} d={self.delta} ●●●")
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
        print(f"\n●●● Testing Model V={self.v} n={self.n} d={self.delta} ●●●")
        # Open test file
        test_set = open(test_path, "r", encoding="utf-8")
        # Create trace file
        trace = open(f"output/trace_{self.v}_{self.n}_{self.delta}.txt", "w", encoding="utf-8")
        for line in test_set.readlines():
            # Checking if string is empty
            # Source: https://www.geeksforgeeks.org/python-program-to-check-if-string-is-empty-or-not/)
            if line and not line.isspace():
                split = line.split("\t")
                tweet_id = split[0]
                language = split[2]
                tweet = pre_process_tweet(self.v, split[3])
                score, predicted_language = self.predict_language(tweet)
                if predicted_language == language:
                    label = "correct"
                else:
                    label = "wrong"
                # Scientific Notation
                # Source: https://kite.com/python/answers/how-to-print-a-number-in-scientific-notation-in-python
                result = f"{tweet_id}  {predicted_language}  {score: .2E}  {language}  {label}\r"
                # print(result)
                trace.write(result)
        trace.close()
        print("...Completed Testing...")

    def evaluate(self):
        print(f"\n●●● Evaluating Model V={self.v} n={self.n} d={self.delta} ●●●")
        # Create evaluation file
        evaluation_file = open(f"output/eval_{self.v}_{self.n}_{self.delta}.txt", "w", encoding="utf-8")
        # Open trace file to evaluate
        trace_file = open(f"output/trace_{self.v}_{self.n}_{self.delta}.txt", "r", encoding="utf-8")

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
            if line and not line.isspace():
                split = line.replace("\n", "").split("  ")
                # Format of trace after split
                # [twitter_id, predicted_lang, score, actual_lang, 'correct/wrong']
                metrics['total'] = metrics.get('total') + 1
                predicted_language = split[1]
                actual_language = split[3]
                if split[4] == 'correct':
                    metrics['correct'] = metrics.get('correct') + 1

                    # The predicted_language is a True Positive
                    metrics[predicted_language]['TP'] = metrics[predicted_language].get('TP') + 1
                    # We don't need to keep track of True Negatives, but it would be that
                    # all other languages (except predicted language) increment in True Negative

                else:   # label = wrong
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
        F1_languages_accounted = 0  # The number of languages with F1 without NaN
        for lang in self.language_probabilities.keys():
            language_metric = metrics.get(lang)
            TP = language_metric.get('TP')
            FP = language_metric.get('FP')
            FN = language_metric.get('FN')

            # Calculate precision TP/(TP+FP)
            if TP+FP > 0:
                P = TP/(TP+FP)
            else:
                P = np.nan
            metrics['Precision'].append(P)

            # Calculate recall TP/(TP+FN)
            if TP+FN > 0:
                R = TP/(TP+FN)
            else:
                R = np.nan
            metrics['Recall'].append(R)
            # Calculate F1 measure (B=1, precision and recall have same importance)
            # F = (B^2 + 1)PR/(B^2P+R)
            B = 1
            if (pow(B, 2)*P+R) > 0:
                F1 = (pow(B, 2) + 1)*P*R/(pow(B, 2)*P+R)
                # Add F1 measure to F1_total and F1_weighted_total
                F1_total += F1
                F1_weighted_total += F1 * self.language_probabilities[lang]
                F1_languages_accounted += 1
            else:
                F1 = np.nan
            metrics['F1'].append(F1)

        evaluation_file.write(f"{metrics['Acc']:.4f}\r")

        precision_formatted = ''
        recall_formatted = ''
        f1_formatted = ''

        for val in metrics['Precision']:
            evaluation_file.write(f"{val:.4f}  ")
            precision_formatted = f"{precision_formatted}{val:>15.4f}"
        evaluation_file.write("\r")

        for val in metrics['Recall']:
            evaluation_file.write(f"{val:.4f}  ")
            recall_formatted = f"{recall_formatted}{val:>15.4f}"
        evaluation_file.write("\r")

        for val in metrics['F1']:
            evaluation_file.write(f"{val:.4f}  ")
            f1_formatted = f"{f1_formatted}{val:>15.4f}"
        evaluation_file.write("\r")

        languages_formatted = ''
        for key in self.language_probabilities.keys():
            languages_formatted = f"{languages_formatted}{key:>15s}"

        number_of_languages = len(self.language_probabilities.keys())
        macro_F1 = F1_total/number_of_languages
        weighted_F1 = F1_weighted_total/number_of_languages

        evaluation_file.write(f"{macro_F1:>.4f}  {weighted_F1:>.4f}\r")

        print(f"{'Languages':15s}{languages_formatted}")
        print(f"{'Precision':15s}{precision_formatted}")
        print(f"{'Recall':15s}{recall_formatted}")
        print(f"{'F1':15s}{f1_formatted}")
        print(f"{'Accuracy':15s}{metrics['Acc']: .4f}")
        print(f"{'Macro-F1':15s}{macro_F1:> .4f}")
        print(f"{'Weighted-F1':15s}{weighted_F1:> .4f}")


