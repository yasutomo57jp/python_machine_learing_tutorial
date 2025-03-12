# WandBの利用

## WandBとは

WandBは、機械学習の実験管理を行うためのツールです。実験の実行結果を記録し、可視化することができます。また、実験の実行状況をリアルタイムで確認することもできます。

## pytorch lightningとの連携

pl.TrainerのコールバックとしてWandbLoggerを指定することで、pytorch lightningとWandBを連携することができます。

https://github.com/yasutomo57jp/python_machine_learing_tutorial/blob/9182bad5885ef18c9c57dfd9eb60bcf3ebb52aa7/2_wandb/train.py#L105:L108

