# Python環境

venvを使ってPython環境を構築する。

環境の作成
python3 -m venv venv

環境の有効化
source ./venv/bin/activate


# SSHの設定

~/.ssh/configに以下を追加する。

```config
Host 自分のPCの名前
  HostName 自分のPCのFQDN や IPアドレス や ホスト名など
```


# 開発環境

VSCodeを使って開発する。

プラグイン
- Remote - SSH
- Python
- Black Formatter
- isort
- GitLens




