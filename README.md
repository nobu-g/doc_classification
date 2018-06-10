# doc_classification
ディープラーニングで文書を「新聞」「雑誌」「教科書」「ブログ」の4クラスに分類します

# model
Bi-LSTM + attention

# Description
- classifier.py : プログラム本体
- bilstm_attention.py : モデル
- train.py : 学習プログラム
- BiLSTM_attention.model : 学習済みモデル

# Dependency
- Python 3.5.1+
- Chainer 4.0.0+
- Juman++
- Word2vec

# Usage
`python classifier.py <分類したい文書>`

classifier.py内のword2vecモデルのパスは適宜書き換えてください
