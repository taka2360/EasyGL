# 🎨 EasyGL - EasyGL - The simple 2D graphics library for Python

**EasyGL**は、Pythonで2Dグラフィックスプログラミングを楽しく直感的に行うために設計されたライブラリです。OpenGLの複雑さをラップし、ゲーム、デモ、ビジュアルプロトタイピングのためのシンプルで高性能なAPIを提供します。

ゲーム開発を学ぶ初心者から、素早くプロトタイプを作りたい専門家まで、EasyGLは最小限のコードであなたのアイデアを形にする手助けをします。

![EasyGL Demo GIF](https://i.imgur.com/your-demo-gif.gif)
*(これはプレースホルダです。`example.py`を動作させたGIFに差し替えることを強く推奨します！)*

---

## ✨ 主な特徴

- **🚀 シンプルで直感的なAPI**: `app.draw_circle()`や`app.draw_line()`のような一行のコードで複雑なシーンを描画できます。
- **⚡ 高パフォーマンス**: インスタンシングやバッチレンダリングといったモダンなOpenGLの技術を利用し、数千個のオブジェクトを滑らかに描画します。
- **🎬 シーン管理**: タイトル画面、ゲーム画面、ゲームオーバー画面など、複数のゲーム状態を管理するためのクリーンなシーンマネージャを内蔵しています。
- **🎨 豊富な描画機能**:
    - **図形**: 矩形、円、楕円、そして太さを調整できる線。
    - **複雑な多角形**: JITコンパイルされた三角形分割アルゴリズムによる、凹型多角形（星形など）のサポート。
    - **画像とテキスト**: PNGやJPG画像を簡単に読み込んで描画し、システムフォントを使ってテキストを表示できます。
- **🖱️ 簡単な入力処理**: キーボードとマウスの入力をシンプルに扱えます。
- **🔢 Zソート**: シンプルな`z`パラメータでオブジェクトの重なり順を制御できます。
- **⏱️ Delta Time**: フレームレートに依存しない物理演算やアニメーションのための、組み込みのデルタタイム追跡機能 (`app.get_delta_time()`)。

---

## 🔧 インストール

EasyGLは、いくつかの一般的なPythonライブラリに依存しています。付属の`requirements.txt`ファイルを使えば、一度にすべてをインストールできます。

```bash
# すべての依存ライブラリをインストール
python -m pip install -r requirements.txt
```

インストール後、メインスクリプトと同じディレクトリに`easy_gl.py`を配置してください。

```
my_awesome_project/
├── easy_gl.py
├── requirements.txt
├── main.py
└── masterpiece.png  # (オプションの画像アセット)
```

---

## 🚀 クイックスタート

跳ねるボールの簡単な例です。

```python
# main.py
from easy_gl import EasyGL

# 1. EasyGLを初期化
app = EasyGL(title="My First App", width=1280, height=720)

# 2. ゲームの状態を定義
ball = {'x': 100, 'y': 100, 'vx': 250, 'vy': 200, 'radius': 30}

# 3. update関数（ロジック）を作成
def update():
    dt = app.get_delta_time() # フレームレートから独立した時間を取得
    
    ball['x'] += ball['vx'] * dt
    ball['y'] += ball['vy'] * dt

    if ball['x'] < ball['radius'] or ball['x'] > 1280 - ball['radius']:
        ball['vx'] *= -1
    if ball['y'] < ball['radius'] or ball['y'] > 720 - ball['radius']:
        ball['vy'] *= -1

# 4. draw関数（描画）を作成
def draw():
    app.draw_circle(ball['x'], ball['y'], ball['radius'], color=(255, 100, 100))
    app.draw_text("My Bouncing Ball!", 20, 20, font="arial", size=24)

# 5. シーンとして追加して実行
app.add_scene("main", setup=None, update=update, draw=draw)
app.run()
```

---

## 🎬 シーン管理

EasyGLでは、アプリケーションを複数のシーン（例：タイトル画面とゲーム画面）に構造化できます。

```python
# --- タイトルシーン ---
def title_update():
    # 1キーでゲームを開始
    if app.is_key_pressed(KEY_1):
        app.set_scene("game") # 'game'シーンに切り替え

def title_draw():
    app.draw_text("My Awesome Game", 400, 300, font="arial", size=48)
    app.draw_text("Press [1] to Start", 480, 400, font="arial", size=24)

# --- ゲームシーン ---
def game_setup():
    # この関数はシーン開始時に一度だけ呼ばれる
    print("Game Started!")

def game_update():
    # ... ゲームのロジック ...
    pass

def game_draw():
    # ... ゲームの描画 ...
    app.draw_circle(640, 360, 50, color=(50, 200, 50))


# --- シーンを登録して実行 ---
app.add_scene("title", setup=None, update=title_update, draw=title_draw)
app.add_scene("game", setup=game_setup, update=game_update, draw=game_draw)

app.set_scene("title") # 初期シーンを設定
app.run()
```

---

## 🎨 APIリファレンス

### 初期化
`EasyGL(title, width, height, max_fps=60)`
- アプリケーションウィンドウを作成・初期化します。

### シーン制御
`app.add_scene(name, setup, update, draw)`
- 新しいシーンを登録します。`setup`はシーン開始時に一度だけ呼ばれるオプションの関数です。
`app.set_scene(name)`
- アクティブなシーンを指定された`name`のシーンに切り替えます。

### コアループ
`app.get_delta_time()`
- 前のフレームからの経過時間を秒単位で返します。滑らかでフレームレートに依存しない動きに不可欠です。

### 描画
*すべての描画関数は、レイヤー分けのためのオプション引数 `z=0` を受け付けます。*

`app.draw_rectangle(x, y, width, height, color)`
`app.draw_circle(x, y, radius, color)`
`app.draw_ellipse(x, y, width, height, color)`
`app.draw_line(x1, y1, x2, y2, thickness, color)`
`app.draw_polygon(points, color)`
- `points`はタプルのリストです: `[(x1, y1), (x2, y2), ...]`

### 画像とテキスト
`app.load_texture(filepath)`
- 画像を読み込み、`tex_id`を返します。
`app.draw_image(tex_id, width, height, x, y, align_center=True)`
- `width`か`height`に`'auto'`を指定すると、アスペクト比を維持します。
`app.draw_text(text, x, y, font, size, color)`

### 入力
`app.is_key_pressed(key)`
- 例: `KEY_A`, `KEY_SPACE`, `KEY_ENTER`。
`app.is_mouse_button_pressed(button)`
- 例: `MOUSE_BUTTON_LEFT`。
`app.get_mouse_position()`
- `(x, y)`座標を返します。

---

## デモ

`example.py`や`stress_test.py`を実行すれば、これらの機能が実際に動作する様子を確認できます。

```bash
python example.py
```
