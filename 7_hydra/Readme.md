# hydraを使ったコードの実行

以下ができていれば，例えば複数のデータセット，データセットの分割に対して，複数の比較手法を実行して比較する，といったことができます．
- ネットワークがパラメータ化されている
- データセットやその分割がパラメータ化されている

その場合，例えば以下の様に実行できます．
```zsh
python train.py dataset=dataset1 network=baseline
```

また，一括して実行する場合，例えば，```baseline1```, ```baseline2```, ```baseline3```を```dataset1```, ```dataset2```で実行する場合
```zsh
for d in {dataset1,dataset2} do
    for n in {baseline1,baseline2,baseline3} do
        python train.py dataset=$d network=$n
    done
done
```
のような実行が可能になります．

