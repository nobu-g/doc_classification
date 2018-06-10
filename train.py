# coding: utf-8
from argparse import ArgumentParser

from gensim.models import KeyedVectors
import numpy as np

import chainer
from chainer import cuda, Variable, optimizers
import chainer.functions as F
from chainer.cuda import to_cpu
from chainer import serializers

from bilstm_attention import BiLSTM


def load_data(path, word2index, n=None):
    with open(path, 'rt') as f:
        lines = f.readlines()

    if n is not None:
        lines = lines[:n]

    source = []
    target = []
    for line in lines:
        words = line[2:].split(" ")
        word_ids = []

        for word in words:
            try:
                word_ids.append(word2index[word])
            except KeyError:
                word_ids.append(word2index['<UNK>'])

        source.append(word_ids)
        target.append(int(line[0]))

    return np.array(source), np.array(target)


# doc(wordidの配列)を受け取りwordidの配列超をlimに制限する。
# 先頭からとってくる時はrandam=False, ランダムな場所からスタートする時はrandam=True
def trim_doc(docs, lim, random):
    trimmed_docs = []
    for doc in docs:
        if len(doc) >= lim:
            if random is False:
                doc = doc[:lim]
                trimmed_docs.append(doc)
            else:
                start_index = np.random.randint(len(doc) - lim + 1)
                doc = doc[start_index : start_index + lim]
                trimmed_docs.append(doc)
        else:
            trimmed_docs.append(doc)

    return np.array(trimmed_docs)


# numpyかcupyの配列([0, 3, 1, 2, ...])を受け取り、マクロf値を計算する。
def macro_f(predict, answer, classes):
    f_sum = 0
    for class_ in range(classes):
        positive = np.sum(predict == class_)
        if not positive == 0:
            precision = np.sum((predict == answer) & (predict == class_)) / positive
        else:
            f_sum += 0
            continue

        true = np.sum(answer == class_)
        if not true == 0:
            recall = np.sum((predict == answer) & (predict == class_)) / true
        else:
            f_sum += 0
            continue

        denom = recall + precision
        if not denom == 0:
            f_sum += (2 * recall * precision) / denom
        else:
            f_sum += 0

    return f_sum / classes


def main():
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--batchsize", default=32, type=int)
    parser.add_argument("--midunits", default=128, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--wordlim", default=512, type=int)
    parser.add_argument("--embedding",
                        default="/share/data/word2vec/2016.08.02/w2v.midasi.256.100K.bin",
                        type=str)
    parser.add_argument("--train",
                        default="/baobab/kiyomaru/2018-shinjin/jumanpp.midasi/train.csv",
                        type=str)
    parser.add_argument("--test",
                        default="/baobab/kiyomaru/2018-shinjin/jumanpp.midasi/test.csv",
                        type=str)
    args = parser.parse_args()

    model_w2v = KeyedVectors.load_word2vec_format(args.embedding, binary=True)
    word2index = {w : i for i, w in enumerate(model_w2v.index2word)}

    # モデルの定義
    model = BiLSTM(mid_size=args.midunits, embed_mat=model_w2v.vectors)
    model.embed.disable_update()

    # GPUを使うかどうか
    gpu_id = args.gpu
    xp = np
    if gpu_id >= 0:
        chainer.backends.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()
        xp = cuda.cupy

    # trainデータとtestデータをロード
    train_x, train_t = load_data(args.train, word2index)
    test_x, test_t = load_data(args.test, word2index)

    # testデータのdocument長をwordlimに制限
    test_x = trim_doc(test_x, lim=args.wordlim, random=False)
    # trainデータも
    train_x = trim_doc(train_x, lim=args.wordlim, random=False)

    test_x = [xp.array(doc, dtype='int32') for doc in test_x]
    test_t = xp.array(test_t, dtype='int32')

    batchsize = args.batchsize
    max_epoch = args.epoch     # 学習エポック数
    N = len(train_x)           # 教師データに含まれる文書の数

    # Setup optimizer
    optimizer = optimizers.MomentumSGD()
    optimizer.setup(model)
#    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0005))

    f_max = 0

    print("epoch          train_loss     test_loss      test_accuracy  test_F")

    # epochループ
    for epoch in range(1, max_epoch + 1):
        perm = np.random.permutation(N)  # ランダムな整数列リストを取得
        loss_train_list = []
        accuracy_train_list = []

        # GPUメモリの都合上、document長をwordlimに制限
#        _train_x = trim_doc(train_x, lim=args.wordlim, random=True)
#        _train_x = np.array(_train_x, dtype="int32")

        # batchループ
        for i in range(0, N, batchsize):
            indices = perm[i : i + batchsize]

            x = [xp.array(doc, dtype='int32') for doc in train_x[indices]]
            t = xp.array(train_t[indices], dtype='int32')

            loss, accuracy = model(x, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            loss_train_list.append(float(loss.data))
            accuracy_train_list.append(float(accuracy.data))

        with chainer.using_config('train', False):
            test_y = model.predict(test_x)[0]
        loss_test = to_cpu(F.softmax_cross_entropy(test_y, test_t).data)
        loss_train = np.mean(loss_train_list)
        accuracy_test = to_cpu(F.accuracy(test_y, test_t).data)
        predict = [np.argmax(vec) for vec in to_cpu(test_y.data)]
        f = macro_f(np.array(predict), to_cpu(test_t), classes=4)

        print("{:0>3} / {:<9}{:<15.6f}{:<15.6f}{:<15.6f}{:<15.6f}"
              .format(epoch, max_epoch, loss_train, loss_test, accuracy_test, f))

        if f > f_max:
            f_max = f
            if f > 0.94:
                serializers.save_npz("BiLSTM_attention.model", model)
                incorrect_list = []
                for i, correct in enumerate(np.array(predict) == to_cpu(test_t)):
                    if not correct:
                        incorrect_list.append(str(i))
                with open("incorrect_doc_num.txt", 'w') as f:
                    f.write(" ".join(incorrect_list))
    print(f_max)


if __name__ == '__main__':
        main()
