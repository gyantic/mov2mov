
このリポジトリは diffusersモジュールを使ってMov2Movを実装したものである。
入力動画は以下のリンクのものを使用させていただいた。
https://pixabay.com/ja/videos/%E7%94%B7-%E3%82%AB%E3%82%A6%E3%83%9C%E3%83%BC%E3%82%A4-%E8%A5%BF%E9%83%A8-%E5%B8%BD%E5%AD%90-208373/


## コードの内容
使用したモデルはSD1.5, SD2, SD3-medium, SD3.5-large　である。
それぞれについてControlNetあるなしでmov2movの実装を行っている。

ファイル名に使用したモデルを記載している。　例えばSD2.1_with_clNetなら、SD2.1についてControlNetを使用したmov2movを実行できるファイルである。




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



ただしSD1.5に限っては、ファインチューニングモデルをダウンロードして使用するので、そのファイルを保存したローカルのパスをpythonファイル内で指定しなければならないことに注意する。
SD1.5のファインチューニングモデルとして使用したもの：https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/blob/main/control_v11p_sd15_canny_fp16.safetensors


詳しい内容は以下の記事で述べている。
