import random
from easy_gl import EasyGL, KEY_UP, KEY_DOWN, KEY_1, KEY_2, KEY_3
import time
import math

# --- 定数 ---
WIDTH, HEIGHT = 1280, 720
INITIAL_OBJECTS = 10000
OBJECT_INCREMENT = 1000

# --- EasyGLの初期化 ---
app = EasyGL(title="Z-Order and Rotation Stress Test", width=WIDTH, height=HEIGHT, max_fps=0)

# --- グローバル変数・リソース ---
objects = []
fps_text_id, object_count_text_id = -1, -1
key_up_pressed, key_down_pressed = False, False
current_test_mode = None # 'simple', 'polygon', or 'mixed'
last_fps_time = 0.0
frame_count = 0

# --- オブジェクトクラス ---
class MovingObject:
    def __init__(self, shape_mode='simple'):
        self.shape_mode = shape_mode
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        size = random.randint(10, 25)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        z = random.choice([0, 1, 2])

        if shape_mode == 'polygon':
            self.shape_type = 'polygon'
            self.num_points = random.randint(3, 7) * 2
            self.outer_radius = size * 1.5
            self.inner_radius = size * 0.7
            self.id = app.draw_polygon([(0,0)], color, z) # Placeholder
        elif shape_mode == 'mixed':
            self.shape_type = random.choice(['circle', 'rectangle', 'polygon'])
            if self.shape_type == 'polygon':
                 self.num_points = random.randint(3, 7) * 2
                 self.outer_radius = size * 1.5
                 self.inner_radius = size * 0.7
                 self.id = app.draw_polygon([(0,0)], color, z)
            elif self.shape_type == 'circle':
                 self.id = app.draw_circle(x, y, size, color, z)
            else:
                 self.id = app.draw_rectangle(x, y, size, size, color, z)
        else: # simple
            self.shape_type = random.choice(['circle', 'rectangle'])
            if self.shape_type == 'circle':
                self.id = app.draw_circle(x, y, size, color, z)
            else:
                self.id = app.draw_rectangle(x, y, size, size, color, z)
        
        app.moveto(self.id, x, y)
        self.vx = random.uniform(-150, 150)
        self.vy = random.uniform(-150, 150)
        self.rotation_speed = random.uniform(-180, 180)

    def update(self, dt):
        obj = app._draw_objects.get(self.id)
        if not obj: return

        new_x = obj['x'] + self.vx * dt
        new_y = obj['y'] + self.vy * dt

        size = obj.get('radius', obj.get('width', 0))
        if new_x - size < 0 or new_x + size > WIDTH: self.vx *= -1
        if new_y - size < 0 or new_y + size > HEIGHT: self.vy *= -1
        
        app.moveto(self.id, new_x, new_y)
        app.rotate(self.id, self.rotation_speed * dt)

# ==============================================================================
#   タイトルシーン
# ==============================================================================

def title_setup():
    app.set_background_color(20, 30, 40)
    app.draw_text("Select Stress Test Mode", WIDTH / 2 - 320, HEIGHT / 2 - 200, font="arial", size=48, color=(255, 255, 255))
    app.draw_text("1: Simple Shapes (Circles & Rectangles) with Rotation", WIDTH / 2 - 300, HEIGHT / 2 - 50, font="arial", size=24, color=(200, 200, 200))
    app.draw_text("2: Complex Shapes (Polygons) with Rotation", WIDTH / 2 - 300, HEIGHT / 2, font="arial", size=24, color=(200, 200, 200))
    app.draw_text("3: Mixed Shapes with Rotation", WIDTH / 2 - 300, HEIGHT / 2 + 50, font="arial", size=24, color=(200, 200, 200))

def title_update():
    global current_test_mode
    if app.is_key_pressed(KEY_1):
        current_test_mode = 'simple'
        app.set_scene("stress_test")
    elif app.is_key_pressed(KEY_2):
        current_test_mode = 'polygon'
        app.set_scene("stress_test")
    elif app.is_key_pressed(KEY_3):
        current_test_mode = 'mixed'
        app.set_scene("stress_test")

def title_draw(): pass

# ==============================================================================
#   ストレステストシーン
# ==============================================================================

def stress_test_setup():
    global objects, last_fps_time, fps_text_id, object_count_text_id
    # Clear previous objects if any
    for obj in objects:
        app.delete_object(obj.id)
    objects.clear()
    app._draw_objects.clear() # Ensure a clean slate

    app.set_background_color(10, 20, 30)
    objects = [MovingObject(shape_mode=current_test_mode) for _ in range(INITIAL_OBJECTS)]
    last_fps_time = time.time()
    
    hud_color = (255, 255, 255)
    font = "arial"
    fps_text_id = app.draw_text(f"FPS: 0", 10, 10, font, 24, hud_color, z=100)
    object_count_text_id = app.draw_text(f"Objects: {len(objects)}", 10, 40, font, 24, hud_color, z=100)
    app.draw_text("UP Arrow: Add Objects", 10, HEIGHT - 60, font, 18, hud_color, z=100)
    app.draw_text("DOWN Arrow: Remove Objects", 10, HEIGHT - 35, font, 18, hud_color, z=100)

def stress_test_update():
    global frame_count, last_fps_time, key_up_pressed, key_down_pressed, fps_text_id, object_count_text_id

    dt = app.get_delta_time()

    frame_count += 1
    current_time = time.time()
    if current_time - last_fps_time >= 1.0:
        app.delete_object(fps_text_id)
        fps_text_id = app.draw_text(f"FPS: {frame_count}", 10, 10, "arial", 24, (255,255,255), z=100)
        frame_count = 0
        last_fps_time = current_time

    for obj in objects:
        obj.update(dt)

    if app.is_key_pressed(KEY_UP) and not key_up_pressed:
        for _ in range(OBJECT_INCREMENT): objects.append(MovingObject(shape_mode=current_test_mode))
        key_up_pressed = True
    elif not app.is_key_pressed(KEY_UP): key_up_pressed = False

    if app.is_key_pressed(KEY_DOWN) and not key_down_pressed:
        for _ in range(OBJECT_INCREMENT):
            if objects:
                obj_to_remove = objects.pop()
                app.delete_object(obj_to_remove.id)
        key_down_pressed = True
    elif not app.is_key_pressed(KEY_DOWN): key_down_pressed = False

    app.delete_object(object_count_text_id)
    object_count_text_id = app.draw_text(f"Objects: {len(objects)}", 10, 40, "arial", 24, (255,255,255), z=100)

def stress_test_draw(): pass

# --- アプリケーションのセットアップと実行 ---
app.add_scene("title", setup=title_setup, update=title_update, draw=title_draw)
app.add_scene("stress_test", setup=stress_test_setup, update=stress_test_update, draw=stress_test_draw)

app.set_scene("title")
app.run()
