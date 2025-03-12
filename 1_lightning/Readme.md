# Pytorch Lightningの利用

MNISTでのpytorch lightningを使った学習コードのサンプルです．


## train.py

LightningModuleを継承したクラスを定義し，そのクラスを使って学習を行います．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/df85028387ee802fdbab2269e29b02e34ec09e2d/1_lightning/train.py#L10-L49

学習に使うデータセットを定義します．．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/df85028387ee802fdbab2269e29b02e34ec09e2d/1_lightning/train.py#L51-L66

プログラムのメイン部分です．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/df85028387ee802fdbab2269e29b02e34ec09e2d/1_lightning/train.py#L69-L87

## train2.py


学習の仕方と，モデルの定義を分離した例です．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/df85028387ee802fdbab2269e29b02e34ec09e2d/1_lightning/train2.py#L10-L22

mainでは，モデルをインスタンス化したうえでlightning moduleへ渡しています．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/df85028387ee802fdbab2269e29b02e34ec09e2d/1_lightning/train2.py#L87-L91
