from src.NaivesBayesClassification import train_model


def main():
    print("Creating models")

    print("===== Creating models for Vocabulary 0 ====")
    print("               == Unigram ==")
    model_v0_unigram = train_model(v=0, n=1, delta=0.5)
    print("               == Bigram ==")
    model_v0_bigram = train_model(v=0, n=2, delta=0.5)
    print("               == Trigram ==")
    model_v0_trigram = train_model(v=0, n=3, delta=0.5)

    print("===== Creating models for Vocabulary 1 ====")
    print("               == Unigram ==")
    model_v1_unigram = train_model(v=1, n=1, delta=0.5)
    print("               == Bigram ==")
    model_v1_bigram = train_model(v=1, n=2, delta=0.5)
    print("               == Trigram ==")
    model_v1_trigram = train_model(v=1, n=3, delta=0.5)

    print("===== Creating models for Vocabulary 1 ====")
    print("               == Unigram ==")
    model_v2_unigram = train_model(v=2, n=1, delta=0.5)
    print("               == Bigram ==")
    model_v2_bigram = train_model(v=2, n=2, delta=0.5)
    print("               == Trigram ==")
    model_v2_trigram = train_model(v=2, n=3, delta=0.5)


if __name__ == "__main__":
    main()
