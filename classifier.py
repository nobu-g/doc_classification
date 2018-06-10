import gensim
import numpy as np
import sys
from pyknp import Jumanpp
import chainer
from chainer import Variable, serializers
import chainer.functions as F
from bilstm_attention import BiLSTM


def doc2list(doc, word2index):
    word_ids = []
    for word in doc:
        try:
            word_ids.append(word2index[word])
        except KeyError:
            word_ids.append(word2index['<UNK>'])

    return np.array(word_ids, dtype='int32')


def main():
    model_w2v = gensim.models.KeyedVectors.load_word2vec_format(
        "/share/data/word2vec/2016.08.02/w2v.midasi.256.100K.bin",
        binary=True, unicode_errors='ignore')
    word2index = {w : i for i, w in enumerate(model_w2v.index2word)}

    model = BiLSTM(embed_mat=model_w2v.vectors, mid_size=128)
    serializers.load_npz("BiLSTM_attention.model", model)

    # 標準入力からテストできるように
    jumanpp = Jumanpp()
    while True:
        input_sentence = sys.stdin.readline()  # 改行を含む, string型
        result = jumanpp.analysis(input_sentence)
        doc = [mrph.midasi for mrph in result.mrph_list()]
        x = [doc2list(doc, word2index)]
#        x = list2Var([doc2vec(doc)], np.float32, False)
        with chainer.using_config("train", False):
            y, attn_list = model.predict(x)

        p = np.argmax(y[0].data)
        doc_class = ["新聞記事", "  雑誌  ", " 教科書 ", " ブログ "]
        print("")
        print("*------------------------*")
        print("|                        |")
        print("|        " + doc_class[p] + "        |")
        print("|                        |")
        print("*------------------------*")
        print("")


        prob = F.softmax(y, axis=1)[0].data
        print("新聞記事: {:.6f}  雑誌: {:.6f}  教科書: {:.6f}  ブログ: {:.6f}".format(prob[0], prob[1], prob[2], prob[3]))

        for word, attn in sorted(zip(doc, attn_list), key=lambda x:x[1], reverse=True):
            print(word, end=", ")
        print("\n")
        
        
if __name__ == "__main__":
    main()
