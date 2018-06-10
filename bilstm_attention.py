import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
import numpy as np


class BiLSTM(Chain):
    def __init__(self, embed_mat, mid_size, out_size=4, dropout=0.25):
        super(BiLSTM, self).__init__()
        with self.init_scope():
            vocab_size, embed_size = embed_mat.shape
            self.embed = L.EmbedID(in_size=vocab_size,
                                   out_size=embed_size,
                                   initialW=embed_mat,
                                   ignore_label=-1)
            self.bi_lstm = L.NStepBiLSTM(n_layers=1,
                                         in_size=embed_size,
                                         out_size=mid_size,
                                         dropout=dropout)
            self.l_attn = L.Linear(mid_size * 2, 1)
            self.l3 = L.Linear(mid_size * 2, out_size)

    def __call__(self, x, t):  # x はid配列(1document)の配列
        # __call__ はlossとaccuracyを返すようにしておく
        y = self.predict(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)

        return loss, accuracy

    def predict(self, x):
        batchsize = len(x)

        xs = [F.dropout(self.embed(Variable(doc)), ratio=0.5) for doc in x]
        hy, cy, ys = self.bi_lstm(hx=None, cx=None, xs=xs)
        # hy: bilstmの最終的な中間層の状態(2, batchsize, mid_size)
        # cy: ??
        # ys: 中間層の各状態のベクトルを保存してある(batchsize, lim, mid_size*2)

        ys = [F.dropout(midvec, ratio=0.3) for midvec in ys]
        concat_ys = F.concat(ys, axis=0)  # (batchsize*lim, mid_size*2)
        attn = F.dropout(self.l_attn(concat_ys), ratio=0.25)  # (batchsize*lim, 1)
        split_attention = F.split_axis(attn, np.cumsum([len(doc) for doc in xs])[:-1], axis=0)  # (batchsize, lim, 1)
        split_attention_pad = F.pad_sequence(split_attention, padding=-1024.0)
        attn_softmax = F.softmax(split_attention_pad, axis=1)  # (batchsize, lim, 1)
        ys_pad = F.pad_sequence(ys, length=None, padding=0.0)  # (batchsize, lim, mid_size*2)

        # ys と attn_softmax の積を計算するためにshapeを揃える
        ys_pad_reshape = F.reshape(ys_pad, (-1, ys_pad.shape[-1]))  # (batchsize*lim, mid_size*2)
        attn_softmax_reshape = F.broadcast_to(F.reshape(attn_softmax, (-1, attn_softmax.shape[-1])), ys_pad_reshape.shape)  # (batchsize*lim, mid_size*2)

        attention_hidden = ys_pad_reshape * attn_softmax_reshape  # 隠れ層 * 重みを計算 (batchsize*lim, mid_size*2)
        attention_hidden_reshape = F.reshape(attention_hidden, (batchsize, -1, attention_hidden.shape[-1]))  # (batchsize*lim, mid_size*2)

        result = F.sum(attention_hidden_reshape, axis=1)  # 隠れ層の重み付き和を計算(batchsize, mid_size*2)

        if chainer.config.train:
            return self.l3(F.dropout(result, ratio=0.2))
#            return self.l3(result)
        else:
            attn_list = F.transpose(attn_softmax[0])[0]
            return self.l3(result), attn_list.data
