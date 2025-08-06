import random
from easy_gl import EasyGL, KEY_UP, KEY_DOWN, KEY_1, KEY_2
import time
import math

# --- 定数 ---
WIDTH, HEIGHT = 1280, 720
INITIAL_OBJECTS = 10000
OBJECT_INCREMENT = 1000

# --- EasyGLの初期化 ---
app = EasyGL(title="Z-Order Stress Test", width=WIDTH, height=HEIGHT, max_fps=0)

# --- グローバル変数・リソース ---
objects = []
fps, frame_count, last_fps_time = 0, 0, 0.0
key_up_pressed, key_down_pressed = False, False
current_test_mode = None # 'simple' or 'polygon'

# --- オブジェクトクラス ---
class MovingObject:
    def __init__(self, shape_mode='simple'):
        if shape_mode == 'polygon':
            self.shape_type = 'polygon'
        else: # simple
            self.shape_type = random.choice(['circle', 'rectangle'])

        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.vx = random.uniform(-150, 150)  # ピクセル/秒
        self.vy = random.uniform(-150, 150)  # ピクセル/秒
        self.size = random.randint(10, 25) if self.shape_type == 'circle' else random.randint(15, 35)
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.z = random.choice([0, 1, 2])

        if self.shape_type == 'polygon':
            self.angle = 0
            self.rotation_speed = random.uniform(-180, 180) # 度/秒
            self.num_points = random.randint(3, 7) * 2
            self.outer_radius = self.size * 1.5
            self.inner_radius = self.size * 0.7

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.shape_type == 'polygon':
            self.angle += self.rotation_speed * dt

        if self.x - self.size < 0 or self.x + self.size > WIDTH: self.vx *= -1
        if self.y - self.size < 0 or self.y + self.size > HEIGHT: self.vy *= -1

    def get_star_points(self):
        points = []
        angle_rad = math.radians(self.angle)
        for i in range(self.num_points):
            r = self.outer_radius if i % 2 == 0 else self.inner_radius
            a = angle_rad - (math.pi / 2) + (i * math.pi / (self.num_points / 2))
            px = self.x + r * math.cos(a)
            py = self.y + r * math.sin(a)
            points.append((px, py))
        return points

# ==============================================================================
#   タイトルシーン
# ==============================================================================

def title_setup():
    app.set_background_color(20, 30, 40)

def title_update():
    global current_test_mode
    if app.is_key_pressed(KEY_1):
        current_test_mode = 'simple'
        app.set_scene("stress_test")
    elif app.is_key_pressed(KEY_2):
        current_test_mode = 'polygon'
        app.set_scene("stress_test")

def title_draw():
    app.draw_text("Select Stress Test Mode", WIDTH / 2 - 320, HEIGHT / 2 - 150, font="arial", size=48, color=(255, 255, 255))
    app.draw_text("1: Simple Shapes (Circles & Rectangles)", WIDTH / 2 - 250, HEIGHT / 2, font="arial", size=24, color=(200, 200, 200))
    app.draw_text("2: Complex Shapes (Polygons)", WIDTH / 2 - 250, HEIGHT / 2 + 50, font="arial", size=24, color=(200, 200, 200))

# ==============================================================================
#   ストレステストシーン
# ==============================================================================

def stress_test_setup():
    global objects, last_fps_time
    app.set_background_color(10, 20, 30)
    objects = [MovingObject(shape_mode=current_test_mode) for _ in range(INITIAL_OBJECTS)]
    last_fps_time = time.time()

def stress_test_update():
    global frame_count, last_fps_time, fps, key_up_pressed, key_down_pressed

    dt = app.get_delta_time()

    # FPS計算
    frame_count += 1
    current_time = time.time()
    if current_time - last_fps_time >= 1.0:
        fps = frame_count
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
            if objects: objects.pop()
        key_down_pressed = True
    elif not app.is_key_pressed(KEY_DOWN): key_down_pressed = False

def stress_test_draw():
    for obj in objects:
        if obj.shape_type == 'circle':
            app.draw_circle(obj.x, obj.y, obj.size, obj.color, z=obj.z)
        elif obj.shape_type == 'rectangle':
            app.draw_rectangle(obj.x, obj.y, obj.size, obj.size, obj.color, z=obj.z)
        elif obj.shape_type == 'polygon':
            app.draw_polygon(obj.get_star_points(), obj.color, z=obj.z)

    hud_color = (255, 255, 255)
    font = "arial"
    app.draw_text(f"FPS: {fps}", 10, 10, font, 24, hud_color, z=100)
    app.draw_text(f"Objects: {len(objects)}", 10, 40, font, 24, hud_color, z=100)
    app.draw_text("UP Arrow: Add Objects", 10, HEIGHT - 60, font, 18, hud_color, z=100)
    app.draw_text("DOWN Arrow: Remove Objects", 10, HEIGHT - 35, font, 18, hud_color, z=100)

# --- アプリケーションのセットアップと実行 ---
app.add_scene("title", setup=title_setup, update=title_update, draw=title_draw)
app.add_scene("stress_test", setup=stress_test_setup, update=stress_test_update, draw=stress_test_draw)

app.set_scene("title")
app.run()
