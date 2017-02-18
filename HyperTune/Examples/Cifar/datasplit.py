#!/usr/bin/env python

import random


if __name__ == '__main__':
    with open("Train_cntk_text.txt", "rb") as f:
        data = f.read().split('\n')

    data = data[:50000]
    print data[49999]

    random.shuffle(data)

    train_data = data[:40000]

    print len(train_data)
    validation_data = data[40000:]

    print len(validation_data)

    with open("Train.txt", "wb") as f1:
        for onedata in train_data:
            f1.write(onedata + '\n')

    with open("Validation.txt", "wb") as f2:
        for onedata in validation_data:
            f2.write(onedata + '\n')

