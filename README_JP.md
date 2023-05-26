# Ninja Lantern

![Ninja Lantern](images/ninja-lantern.jpg)

Ninja Lanternは、手の印に反応して色を変えることができるインタラクティブなトルコランプです。Pythonと機械学習を使用して忍者の印を認識し、それに対応する色にランプの光を変更します。

## 機能

- 忍者の手印をカメラでリアルタイムに認識
- 手印に対応する色をRGB LEDで表示
- 無限の色の可能性と組み合わせ

## 必要なハードウェア

- Raspberry Pi (with a USB Camera)
- RGB LED
- 適切な抵抗とワイヤー

## セットアップ

1. 必要なハードウェアを集め、Raspberry Piに接続します。
2. Raspberry PiのOSに必要なPythonライブラリをインストールします。
3. このリポジトリをクローンまたはダウンロードします。

```bash
git clone https://github.com/username/ninja-lantern.git
```

4. `ninja-lantern` ディレクトリに移動します。

```bash
cd ninja-lantern
```

5. スクリプトを実行してNinja Lanternを起動します。

```bash
python3 main.py
```

## 使い方

手印をカメラの前に持っていき、ランプがそれに応じて色を変えるのを楽しんでください。印と色のマッピングは、`config.py` ファイルでカスタマイズすることができます。

## 貢献

フィードバックやプルリクエストは常に歓迎しています。

## ライセンス

[MIT License](LICENSE)
