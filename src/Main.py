from src.Model import Model
from os import path


def main():
    print("⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆")
    print("★ Welcome to Language Detection by No Look Pass ★")
    print("⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆")
    while True:
        print("Enter Training Set file path: ", end="")
        training_set = input()
        training_set = "OriginalDataSet/training-tweets.txt"
        # Source: https://www.guru99.com/python-check-if-file-exists.html
        if path.exists(training_set):
            print("\tOK: Training File Accepted\n")
            break
        else:
            print("\tERROR: File path does not exist\n")

    while True:
        print("Enter Test Set file path: ", end="")
        test_set = input()
        test_set = "OriginalDataSet/test1.txt"
        # Source: https://www.guru99.com/python-check-if-file-exists.html
        if path.exists(test_set):
            print("\tOK: Test File Accepted")
            break
        else:
            print("\tERROR: File path does not exist\n")

    # Build models using training set

    v_0_model = Model(v=0, n=1, delta=0.5, training_path=training_set)
    v_1_model = Model(v=1, n=1, delta=0.5, training_path=training_set)
    # v_2_model = Model(v=2, n=1, delta=0.5, training_path=training_set)

    # Test models using test set

    v_0_model.test(test_path=test_set)
    v_1_model.test(test_path=test_set)
    # v_2_model.test(test_path=test_set)

    v_0_model.evaluate()
    v_1_model.evaluate()


if __name__ == "__main__":
    main()


