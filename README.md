# EasyGL: シンプルな2Dグラフィックライブラリ for Python

`EasyGL` は、OpenGLの複雑さをラップし、Pythonで手軽に2Dグラフィックスを扱えるように設計されたライブラリです。デモやプロトタイピング、簡単なゲーム制作などに適しています。

内部的には、大量のオブジェクトを効率的に描画するためのインスタンシングやバッチ処理などの最適化技術を使用しています。

![EasyGL Demo](https://i.imgur.com/example.png)  
*(注: 上の画像はデモのイメージです。実際の `example.py` のスクリーンショットをここに挿入することを推奨します。)*

---

## ◆ 主な機能

- **シンプルなAPI:** `draw_rectangle`, `draw_circle` のような直感的な関数で描画できます。
- **多彩な図形描画:**
    - 矩形 (`draw_rectangle`)
    - 円 (`draw_circle`)
    - 楕円 (`draw_ellipse`)
    - 凹形状にも対応した多角形 (`draw_polygon`)
- **画像とテキスト:**
    - 画像ファイルの読み込みと描画 (`load_texture`, `draw_image`)
    - アスペクト比を維持した画像の自動リサイズ (`width='auto'` / `height='auto'`)
    - システムフォントを使用したテキスト描画 (`draw_text`)
- **パフォーマンス:**
    - **インスタンシング:** 矩形や円など、同じ形状のオブジェクトを一度の命令で大量に描画します。
    - **バッチ処理:** テクスチャやポリゴンをグループ化し、描画呼び出しの回数を削減します。
- **入力ハンドリング:** キーボードとマウスの入力を簡単に扱えます。
- **Zソート:** `z`値を指定することで、オブジェクトの重なり順を制御できます。

---

## ◆ セットアップ

### 必要なライブラリ

EasyGLを使用するには、以下のライブラリが必要です。

```bash
# pipを使用してインストール
python -m pip install glfw pyopengl numpy pillow
```

### ファイル構成

プロジェクトフォルダに `easy_gl.py` と、それを使用するあなたのスクリプト（例: `main.py`）を配置します。

```
my_project/
├── easy_gl.py
├── main.py
└── test_image.png (オプション)
```

---

## ◆ 基本的な使い方

EasyGLの基本的な構造は、`update`（状態更新）と`draw`（描画）の2つの関数をループで回すことです。

```python
# main.py
from easy_gl import EasyGL, KEY_W, KEY_S

# 1. EasyGLのインスタンスを作成
app = EasyGL(title="My App", width=1280, height=720, max_fps=60)

# 2. ゲームの状態を定義
player_x, player_y = 640, 360

# 3. 状態を更新する関数を定義
def update():
    global player_y
    if app.is_key_pressed(KEY_W):
        player_y -= 5
    if app.is_key_pressed(KEY_S):
        player_y += 5

# 4. 描画する関数を定義
def draw():
    # (x, y, 半径, 色)
    app.draw_circle(player_x, player_y, 30, color=(255, 100, 100))
    
    # (テキスト, x, y, フォント名, サイズ, 色)
    app.draw_text("Use W/S keys to move!", 20, 20, font="arial", size=24, color=(255, 255, 255))

# 5. 関数を登録して実行
app.set_update_function(update)
app.set_draw_function(draw)
app.run()
```

---

## ◆ APIリファレンス

### 図形描画

`draw_rectangle(x, y, width, height, color=(255,255,255), z=0)`
- 中心座標 `(x, y)` に矩形を描画します。

`draw_circle(x, y, radius, color=(255,255,255), z=0)`
- 中心座標 `(x, y)` に円を描画します。

`draw_ellipse(x, y, width, height, color=(255,255,255), z=0)`
- 中心座標 `(x, y)` に楕円を描画します。

`draw_polygon(points, color=(255,255,255), z=0)`
- 頂点のリスト `points` から多角形を描画します。星形のような凹んだ形状にも対応しています。
- `points`: `[(x1, y1), (x2, y2), ...]` の形式。

### 画像とテキスト

`load_texture(filepath)`
- 画像ファイルを読み込み、テクスチャIDを返します。このIDは `draw_image` で使用します。

`draw_image(tex_id, width, height, x, y, align_center=True, z=0)`
- 指定されたテクスチャIDの画像を描画します。
- `width` / `height`: `'auto'` を指定すると、アスペクト比を維持して自動リサイズします。
    - `width=150, height='auto'`: 幅を150に固定し、高さを自動調整。
    - `width='auto', height='auto'`: オリジナルサイズで描画。
- `align_center`: `True` の場合、`(x, y)` が画像の中心になるように描画します。

`draw_text(text, x, y, font, size, color=(255,255,255), z=0)`
- テキストを描画します。
- `font`: `'arial'`, `'times new roman'` のように、システムにインストールされているフォント名を指定します。

### 入力

`is_key_pressed(key)`
- 指定されたキーが押されているか (`True`/`False`) を返します。
- `key`: `KEY_A`, `KEY_SPACE` などの定数を指定します。

`is_mouse_button_pressed(button)`
- 指定されたマウスボタンが押されているか (`True`/`False`) を返します。
- `button`: `MOUSE_BUTTON_LEFT`, `MOUSE_BUTTON_RIGHT` などを指定します。

`get_mouse_position()`
- 現在のマウスカーソルの `(x, y)` 座標を返します。

---

## ◆ デモ

`example.py` を実行すると、このライブラリのほぼ全ての機能を使ったデモを見ることができます。多数のオブジェクトが滑らかに動く様子や、図形、画像、テキストの描画、入力への反応などを確認できます。

```bash
python example.py
```
