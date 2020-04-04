from src.Model import Model
from os import path, makedirs


# Source: https://www.tutorialspoint.com/How-can-I-create-a-directory-if-it-does-not-exist-using-Python
def create_directory(folder_path):
    if not path.exists(folder_path):
        makedirs(folder_path)


def main():
    print("⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆")
    print("★ Welcome to Language Detection by No Look Pass ★")
    print("⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆⋆")

    training_set = "OriginalDataSet/training-tweets.txt"

    # Training Set is fixed for the purposes of the demo
    # while True:
    #     print("Enter Training Set file path: ", end="")
    #     training_set = input()
    #     training_set = "OriginalDataSet/training-tweets.txt"
    #     # Source: https://www.guru99.com/python-check-if-file-exists.html
    #     if path.exists(training_set):
    #         print("\tOK: Training File Accepted\n")
    #         break
    #     else:
    #         print("\tERROR: File path does not exist\n")

    while True:
        print("Enter Test Set file path: ", end="")
        test_set = input()
        # test_set = "OriginalDataSet/test-tweets-given.txt"
        # Source: https://www.guru99.com/python-check-if-file-exists.html
        if path.exists(test_set):
            print("\tOK: Test File Accepted\n")
            break
        else:
            print("\tERROR: File path does not exist\n")

    # Create output folder
    create_directory("output/")

    # Initialize models
    byom = Model(v=3, n=3, delta=0.1)
    model_v0_n1_d0 = Model(v=0, n=1, delta=0)
    model_v1_n2_d05 = Model(v=1, n=2, delta=0.5)
    model_v1_n3_d1 = Model(v=1, n=3, delta=1)
    model_v2_n2_d03 = Model(v=2, n=2, delta=0.3)

    # Train models
    model_v0_n1_d0.train(training_set)
    model_v1_n2_d05.train(training_set)
    model_v1_n3_d1.train(training_set)
    model_v2_n2_d03.train(training_set)
    byom.train(training_set)

    # Test models using test set
    model_v0_n1_d0.test(test_set)
    model_v1_n2_d05.test(test_set)
    model_v1_n3_d1.test(test_set)
    model_v2_n2_d03.test(test_set)
    byom.test(test_set)

    # Evaluate models
    model_v0_n1_d0.evaluate()
    model_v1_n2_d05.evaluate()
    model_v1_n3_d1.evaluate()
    model_v2_n2_d03.evaluate()
    byom.evaluate()


if __name__ == "__main__":
    main()



