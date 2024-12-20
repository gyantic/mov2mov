## mov2mov

このリポジトリは diffusersモジュールを使ってMov2Movを実装したものです



## コードの内容
使用したモデルはSD1.5, SD2, SD3-medium, SD3.5-large　である。
それぞれについてControlNetあるなしでmov2movの実装を行っている。

mov2mov_sd1.5_with_clNet.pyとmov2mov_controlNet.pyがSD1.5の実写モデル()を使用して実装したmov2movの実行ファイルである。




## 使用方法
mov2mov関数を実行することでmov2movがなされるようになっている。
mov2mov関数の引数は、
入力動画（mov2movしたい動画）、入力動画のフレーム保存場所、出力動画の保存先
とする。
例えば
```
mov2mov("input.mp4, frames, "output.mp4")
```
とすればいい。



ただしSD1.5に限っては、ファインチューニングモデルをダウンロードして使用するのでそのファイルを保存したローカルのパスを指定しなければならない。



詳しい内容は以下の記事で述べている。
