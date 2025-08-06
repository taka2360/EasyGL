import random
import time
import math
from easy_gl import EasyGL, KEY_W, KEY_A, KEY_S, KEY_D, MOUSE_BUTTON_LEFT

# --- 定数 ---
WIDTH, HEIGHT = 1600, 900
NUM_OBJECTS = 5000
PLAYER_SPEED = 5

# --- EasyGLの初期化 ---
app = EasyGL(title="EasyGL Features Demo", width=WIDTH, height=HEIGHT, max_fps=0)
app.set_background_color(10, 20, 30)

# --- ゲームの状態管理 ---
player_pos = [WIDTH / 2, HEIGHT / 2]
objects = []

# FPS計算用
frame_count = 0
last_fps_update_time = time.time()
fps_display = "FPS: -"

# 回転する星（ポリゴン）用
star_angle = 0

# --- リソースの読み込み ---
try:
    test_texture = app.load_texture(r"C:\Users\yuasa\gemini_cli\modules\test_masterpiece.png")
except FileNotFoundError:
    test_texture = -1
    print("Warning: test_image.png not found.")

def get_star_points(cx, cy, outer_radius, inner_radius, points, angle_deg):
    rotated_points = []
    angle_rad = math.radians(angle_deg)
    for i in range(points * 2):
        r = outer_radius if i % 2 == 0 else inner_radius
        a = angle_rad - (math.pi / 2) + (i * math.pi / points)
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        rotated_points.append((x, y))
    return rotated_points

def setup():
    for _ in range(NUM_OBJECTS):
        objects.append([
            random.uniform(0, WIDTH), random.uniform(0, HEIGHT),
            random.uniform(-2, 2), random.uniform(-2, 2),
            random.uniform(5, 15), 
            random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), random.uniform(0.2, 1.0)
        ])

def update():
    global player_pos, frame_count, last_fps_update_time, fps_display, star_angle

    frame_count += 1
    current_time = time.time()
    if current_time - last_fps_update_time >= 1.0:
        fps_display = f"FPS: {frame_count}"
        frame_count = 0
        last_fps_update_time = current_time

    if app.is_key_pressed(KEY_W): player_pos[1] -= PLAYER_SPEED
    if app.is_key_pressed(KEY_S): player_pos[1] += PLAYER_SPEED
    if app.is_key_pressed(KEY_A): player_pos[0] -= PLAYER_SPEED
    if app.is_key_pressed(KEY_D): player_pos[0] += PLAYER_SPEED

    for obj in objects:
        obj[0] += obj[2]
        obj[1] += obj[3]
        if obj[0] - obj[4] < 0 or obj[0] + obj[4] > WIDTH: obj[2] *= -1
        if obj[1] - obj[4] < 0 or obj[1] + obj[4] > HEIGHT: obj[3] *= -1

    star_angle += 1

def draw():
    # 背景オブジェクト
    for x, y, _, _, size, r, g, b in objects:
        app.draw_circle(x, y, size, color=(int(r*255), int(g*255), int(b*255)))

    # 前景オブジェクト (z=10)
    z_foreground = 10
    app.draw_rectangle(player_pos[0], player_pos[1], 50, 50, color=(50, 200, 50), z=z_foreground)
    mouse_x, mouse_y = app.get_mouse_position()
    app.draw_ellipse(mouse_x, mouse_y, 60, 30, color=(200, 100, 255), z=z_foreground)
    star_points = get_star_points(WIDTH / 2, 100, 80, 40, 5, star_angle)
    app.draw_polygon(star_points, color=(255, 223, 0), z=z_foreground)

    # --- 画像描画のデモ (z=10) ---
    if test_texture != -1:
        # 1. 幅を固定して高さを自動調整
        app.draw_image(test_texture, width=150, height='auto', x=WIDTH - 180, y=100, align_center=True, z=z_foreground)
        app.draw_text("W:150, H:auto", WIDTH - 180, 180, font="arial", size=14, color=(255,255,255), z=z_foreground+1)

        # 2. 高さを固定して幅を自動調整
        app.draw_image(test_texture, width='auto', height=100, x=WIDTH - 180, y=280, align_center=True, z=z_foreground)
        app.draw_text("W:auto, H:100", WIDTH - 180, 340, font="arial", size=14, color=(255,255,255), z=z_foreground+1)


    if app.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
        mx, my = app.get_mouse_position()
        app.draw_circle(mx, my, 20, color=(255, 255, 0), z=z_foreground)
        app.draw_text(f"({int(mx)}, {int(my)})", mx + 25, my - 10, font="arial", size=16, color=(255, 255, 255), z=z_foreground)

    # UI (z=20)
    z_ui = 20
    app.draw_text(f"FPS: {fps_display}", 10, 10, font="arial", size=24, color=(255, 255, 255), z=z_ui)
    app.draw_text(f"Objects: {NUM_OBJECTS}", 10, 40, font="arial", size=24, color=(255, 255, 255), z=z_ui)
    app.draw_text("Image sizing demo on the right", 10, 70, font="arial", size=18, color=(200, 200, 200), z=z_ui)

# --- 実行 ---
setup()
app.set_update_function(update)
app.set_draw_function(draw)
app.run()