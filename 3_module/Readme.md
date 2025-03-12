# コードのモジュール化

ネットワークをモジュールとして切り出して，それを使って学習するコードの例．

まず，ネットワークを別のファイルに切り出します．  
このとき，必要なモジュールをインポートしておきます．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/eed43de6e46e04dbd204193de8357e13935aef47/3_module/networks/baseline.py#L1-L17


networks/__init__.py でインポートしておくと，

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/eed43de6e46e04dbd204193de8357e13935aef47/3_module/networks/__init__.py#L1-L2

使う側では，以下のようにインポートして使えます．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/eed43de6e46e04dbd204193de8357e13935aef47/3_module/train.py#L6




