# コードのモジュール化

## ネットワークのモジュール化

まず，ネットワークを別のファイルに切り出します．  
このとき，必要なモジュールをインポートしておきます．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/eed43de6e46e04dbd204193de8357e13935aef47/3_module/networks/baseline.py#L1-L17

これを，networksディレクトリに保存します．


```networks/__init__.py``` でインポートしておくと，

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/eed43de6e46e04dbd204193de8357e13935aef47/3_module/networks/__init__.py#L1-L2

使う側では，以下のようにインポートして使えます．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/eed43de6e46e04dbd204193de8357e13935aef47/3_module/train.py#L6


## datasetのモジュール化
また，同様にデータセットを別のファイルに切り出します．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/12cdf242c07374cd109fb8492dc19063b02447d3/3_module/dataset.py#L1-L30

## Testのコード
モジュール化したので，テスト時もそれを読み込み，学習済み重みを読み込んでテストします．

データセットモジュールにテスト用のDataLoaderを追加しています．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/12cdf242c07374cd109fb8492dc19063b02447d3/3_module/dataset.py#L21

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/12cdf242c07374cd109fb8492dc19063b02447d3/3_module/dataset.py#L29-L30

テストしたいモデルの読み込みと学習済みモデルの読み込みます．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/2578f27a7555a0675fd047c1419c227d91c4e400/3_module/test.py#L17-L19

テスト時は，```pl.Trainer```のtestを呼びます．

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/12cdf242c07374cd109fb8492dc19063b02447d3/3_module/test.py#L29-L31
