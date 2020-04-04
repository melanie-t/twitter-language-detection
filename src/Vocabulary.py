def build_vocab0(n):
    vocabulary = dict()
    a = 97
    if n == 1:
        for i in range(0, 26):
            vocabulary[chr(a+i)] = 0
    elif n == 2:
        for i in range(0, 26):
            for j in range(0, 26):
                vocabulary[chr(a+i)+chr(a+j)] = 0
                # vocabulary[chr(a+i)+"|"+chr(a+j)] = 0
    elif n == 3:
        for i in range(0, 26):
            for j in range(0, 26):
                for k in range(0, 26):
                    vocabulary[chr(a+i)+chr(a+j)+chr(a+k)] = 0
    return vocabulary


def build_vocab1(n):
    vocabulary = dict()
    A = 65
    a = 97

    if n == 1:
        for i in range(0, 26):
            vocabulary[chr(A+i)] = 0
            vocabulary[chr(a+i)] = 0

    elif n == 2:
        for i in range(0, 26):
            for j in range(0, 26):
                vocabulary[chr(A+i)+chr(A+j)] = 0        # 1: All upper-case
                vocabulary[chr(a+i)+chr(a+j)] = 0        # 2: All lower-case
                vocabulary[chr(A+i)+chr(a+j)] = 0        # 3: First char upper, second char lower
                vocabulary[chr(a+i)+chr(A+j)] = 0        # 4: First char lower, second char upper

    elif n == 3:
        for i in range(0, 26):
            for j in range(0, 26):
                for k in range(0, 26):
                    vocabulary[chr(A+i)+chr(A+j)+chr(A+k)] = 0       # 1: All upper case (UUU)
                    vocabulary[chr(a+i)+chr(a+j)+chr(a+k)] = 0       # 5: All lower case (LLL)
                    vocabulary[chr(a+i)+chr(a+j)+chr(A+k)] = 0       # 6: Lower, Lower, Upper (LLU)
                    vocabulary[chr(A+i)+chr(A+j)+chr(a+k)] = 0       # 2: Upper, Upper, Lower (UUL)
                    vocabulary[chr(A+i)+chr(a+j)+chr(A+k)] = 0       # 3: Upper, Lower, Upper (ULU)
                    vocabulary[chr(a+i)+chr(A+j)+chr(A+k)] = 0       # 4: Lower, Upper, Upper (LUU)
                    vocabulary[chr(a+i)+chr(A+j)+chr(a+k)] = 0       # 7: Lower, Upper, Lower (LUL)
                    vocabulary[chr(A+i)+chr(a+j)+chr(a+k)] = 0       # 8: Upper, Lower, Lower (ULL)
    return vocabulary
