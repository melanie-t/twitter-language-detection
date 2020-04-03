from src.Model import Model
from src.NaivesBayesClassification import train_model


def main():
    training_set = "OriginalDataSet/training-tweets.txt"
    test_set = "OriginalDataSet/test1.txt"

    # Build models using training set
    print("{:<50s}".format("●●● Creating model for Vocabulary 0 ●●●"))
    v_0_model = Model(v=0, n=1, delta=0.5, training_path=training_set)

    print("{:<50s}".format("●●● Creating model for Vocabulary 1 ●●●"))
    v_1_model = Model(v=1, n=1, delta=0.5, training_path=training_set)

    print("{:<50s}".format("●●● Creating model for Vocabulary 1 ●●●"))
    v_2_model = Model(v=2, n=1, delta=0.5, training_path=training_set)

    # Test models using test set
    v_0_model.test(test_path=test_set)
    v_1_model.test(test_path=test_set)
    v_2_model.test(test_path=test_set)


if __name__ == "__main__":
    main()


