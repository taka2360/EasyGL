# 🎨 EasyGL - The simple 2D graphics library for Python

**EasyGL** is a Python library designed to make 2D graphics programming fun and intuitive. It wraps the complexities of OpenGL, providing a simple, high-performance API for creating games, demos, and visual prototypes.

Whether you're a beginner learning gamedev or an expert building a quick prototype, EasyGL helps you bring your ideas to life with minimal code.

![EasyGL Demo GIF](https://i.imgur.com/your-demo-gif.gif)
*(This is a placeholder. It's highly recommended to replace this with a GIF of `example.py` in action!)*

---

## ✨ Features

- **🚀 Simple & Intuitive API**: Draw complex scenes with one-liners like `app.draw_circle()` and `app.draw_line()`.
- **⚡ High Performance**: Utilizes modern OpenGL features like instancing and batch rendering to draw thousands of objects smoothly.
- **🎬 Scene Management**: Built-in support for managing multiple game states (e.g., title screen, game, game over) with a clean scene manager.
- **🎨 Rich Drawing Capabilities**:
    - **Shapes**: Rectangles, Circles, Ellipses, and Lines with adjustable thickness.
    - **Complex Polygons**: Concave polygon support (e.g., stars) powered by a JIT-compiled triangulation algorithm.
    - **Images & Text**: Easily load and draw images (PNG, JPG) and text using system fonts.
- **🖱️ Effortless Input**: Simple methods to handle keyboard and mouse input.
- **🔢 Z-Ordering**: Control the layering of objects with a simple `z` parameter.
- **⏱️ Delta Time**: Built-in delta time tracking (`app.get_delta_time()`) for frame-rate independent physics and animations.

---

## 🔧 Installation

EasyGL requires a few common Python libraries. You can install them all at once using the provided `requirements.txt` file:

```bash
# Install all dependencies
python -m pip install -r requirements.txt
```

Then, simply place `easy_gl.py` in your project directory alongside your main script.

```
my_awesome_project/
├── easy_gl.py
├── main.py
└── masterpiece.png  # (Optional image asset)
```

---

## 🚀 Quick Start

Here's a simple example of a bouncing ball.

```python
# main.py
from easy_gl import EasyGL

# 1. Initialize EasyGL
app = EasyGL(title="My First App", width=1280, height=720)

# 2. Define your game state
ball = {'x': 100, 'y': 100, 'vx': 250, 'vy': 200, 'radius': 30}

# 3. Create the update function (logic)
def update():
    dt = app.get_delta_time() # Get frame-rate independent time
    
    ball['x'] += ball['vx'] * dt
    ball['y'] += ball['vy'] * dt

    if ball['x'] < ball['radius'] or ball['x'] > 1280 - ball['radius']:
        ball['vx'] *= -1
    if ball['y'] < ball['radius'] or ball['y'] > 720 - ball['radius']:
        ball['vy'] *= -1

# 4. Create the draw function (rendering)
def draw():
    app.draw_circle(ball['x'], ball['y'], ball['radius'], color=(255, 100, 100))
    app.draw_text("My Bouncing Ball!", 20, 20, font="arial", size=24)

# 5. Add as a scene and run!
app.add_scene("main", setup=None, update=update, draw=draw)
app.run()
```

---

## 🎬 Scene Management

EasyGL allows you to structure your application into different scenes (like a title screen and a game screen).

```python
# --- Title Scene ---
def title_update():
    # Press 1 to start the game
    if app.is_key_pressed(KEY_1):
        app.set_scene("game") # Switch to the 'game' scene

def title_draw():
    app.draw_text("My Awesome Game", 400, 300, font="arial", size=48)
    app.draw_text("Press [1] to Start", 480, 400, font="arial", size=24)

# --- Game Scene ---
def game_setup():
    # This function is called once when the scene starts
    print("Game Started!")

def game_update():
    # ... game logic here ...
    pass

def game_draw():
    # ... game drawing here ...
    app.draw_circle(640, 360, 50, color=(50, 200, 50))


# --- Register scenes and run ---
app.add_scene("title", setup=None, update=title_update, draw=title_draw)
app.add_scene("game", setup=game_setup, update=game_update, draw=game_draw)

app.set_scene("title") # Set the initial scene
app.run()
```

---

## 🎨 API Reference

### Initialization
`EasyGL(title, width, height, max_fps=60)`
- Creates and initializes the application window.

### Scene Control
`app.add_scene(name, setup, update, draw)`
- Registers a new scene. `setup` is an optional function called once on scene start.
`app.set_scene(name)`
- Switches the active scene to the one with the given `name`.

### Core Loop
`app.get_delta_time()`
- Returns the time in seconds since the last frame. Essential for smooth, frame-rate independent movement.

### Drawing
*All draw functions accept an optional `z=0` argument for layering.*

`app.draw_rectangle(x, y, width, height, color)`
`app.draw_circle(x, y, radius, color)`
`app.draw_ellipse(x, y, width, height, color)`
`app.draw_line(x1, y1, x2, y2, thickness, color)`
`app.draw_polygon(points, color)`
- `points` is a list of tuples: `[(x1, y1), (x2, y2), ...]`

### Images & Text
`app.load_texture(filepath)`
- Loads an image and returns a `tex_id`.
`app.draw_image(tex_id, width, height, x, y, align_center=True)`
- `width` or `height` can be `'auto'` to preserve the aspect ratio.
`app.draw_text(text, x, y, font, size, color)`

### Input
`app.is_key_pressed(key)`
- e.g., `KEY_A`, `KEY_SPACE`, `KEY_ENTER`.
`app.is_mouse_button_pressed(button)`
- e.g., `MOUSE_BUTTON_LEFT`.
`app.get_mouse_position()`
- Returns `(x, y)`.

---

## demos

Check out `example.py` and `stress_test.py` to see all these features in action!

```bash
python example.py
```