import random
import time
import math
from easy_gl import EasyGL, KEY_W, KEY_A, KEY_S, KEY_D, MOUSE_BUTTON_LEFT, KEY_ENTER

# --- 定数 ---
WIDTH, HEIGHT = 1600, 900
NUM_OBJECTS = 5000
PLAYER_SPEED = 5

# --- EasyGLの初期化 ---
app = EasyGL(title="EasyGL Scene Demo", width=WIDTH, height=HEIGHT, max_fps=0)

# --- グローバル変数・リソース ---
player_id = -1
objects = []
test_texture_id = -1
star_id = -1
fps_text_id = -1
info_text_id = -1
image_info_text1_id = -1
image_info_text2_id = -1
line_id = -1
line_text_id = -1
click_circle_id = -1
click_text_id = -1
title_text_id = -1
prompt_text_id = -1
test_image1_id = -1
test_image2_id = -1
player_ellipse_id = -1

# ==============================================================================
#   タイトルシーン
# ==============================================================================

def title_setup():
    global title_text_id, prompt_text_id
    app.set_background_color(20, 30, 40)
    title_text_id = app.draw_text("EasyGL Scene Demo", WIDTH / 2 - 250, HEIGHT / 2 - 100, font="arial", size=48, color=(255, 255, 255), z=10)
    prompt_text_id = app.draw_text("Press ENTER to Start", WIDTH / 2 - 150, HEIGHT / 2 + 20, font="arial", size=24, color=(200, 200, 200), z=10)

def title_update():
    if app.is_key_pressed(KEY_ENTER):
        app.delete_object(title_text_id)
        app.delete_object(prompt_text_id)
        app.set_scene("game")

def title_draw():
    pass # オブジェクトはsetupで作成済

# ==============================================================================
#   ゲームシーン
# ==============================================================================

def game_setup():
    global player_id, objects, test_texture_id, star_id, fps_text_id, info_text_id, last_fps_update_time
    global image_info_text1_id, image_info_text2_id, line_id, line_text_id, test_image1_id, test_image2_id, player_ellipse_id
    app.set_background_color(10, 20, 30)
    
    # オブジェクトの生成
    objects = []
    for _ in range(NUM_OBJECTS):
        obj_id = app.draw_circle(
            random.uniform(0, WIDTH), random.uniform(0, HEIGHT),
            random.uniform(5, 15),
            color=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        )
        objects.append({
            'id': obj_id,
            'vx': random.uniform(-2, 2),
            'vy': random.uniform(-2, 2)
        })

    # プレイヤーと他のオブジェクト
    player_id = app.draw_rectangle(WIDTH / 2, HEIGHT / 2, 50, 50, color=(50, 200, 50), z=10)
    player_ellipse_id = app.draw_ellipse(0, 0, 60, 30, color=(200, 100, 255), z=10)
    star_id = app.draw_polygon([(0,0)], color=(255, 223, 0), z=10) # Placeholder
    
    try:
        test_texture_id = app.load_texture("test_masterpiece.png")
    except Exception as e:
        test_texture_id = -1
        print(f"Warning: test_masterpiece.png not found. {e}")

    if test_texture_id != -1:
        test_image1_id = app.draw_image(test_texture_id, width=150, height='auto', x=WIDTH - 180, y=100, align_center=True, z=10)
        image_info_text1_id = app.draw_text("W:150, H:auto", WIDTH - 220, 180, font="arial", size=14, color=(255,255,255), z=11)
        test_image2_id = app.draw_image(test_texture_id, width='auto', height=100, x=WIDTH - 180, y=280, align_center=True, z=10)
        image_info_text2_id = app.draw_text("W:auto, H:100", WIDTH - 220, 340, font="arial", size=14, color=(255,255,255), z=11)

    # UI要素
    fps_text_id = app.draw_text("FPS: -", 10, 10, font="arial", size=24, color=(255, 255, 255), z=20)
    info_text_id = app.draw_text(f"Objects: {NUM_OBJECTS}", 10, 40, font="arial", size=24, color=(255, 255, 255), z=20)
    app.draw_text("Image sizing demo on the right", 10, 70, font="arial", size=18, color=(200, 200, 200), z=20)
    
    line_id = app.draw_line(0,0,0,0, thickness=5, color=(100, 150, 255), z=11)
    line_text_id = app.draw_text("Line Drawing Demo", WIDTH / 2 - 80, HEIGHT - 30, font="arial", size=16, color=(100, 150, 255), z=12)

    last_fps_update_time = time.time()

def game_update():
    global last_fps_update_time, frame_count, click_circle_id, click_text_id, star_id, line_id
    
    # FPSカウンター
    frame_count += 1
    current_time = time.time()
    if current_time - last_fps_update_time >= 1.0:
        app.delete_object(fps_text_id)
        app.draw_text(f"FPS: {frame_count}", 10, 10, font="arial", size=24, color=(255, 255, 255), z=20)
        frame_count = 0
        last_fps_update_time = current_time

    # プレイヤーの移動と回転
    player_obj = app._draw_objects[player_id]
    if app.is_key_pressed(KEY_W): player_obj['y'] -= PLAYER_SPEED
    if app.is_key_pressed(KEY_S): player_obj['y'] += PLAYER_SPEED
    if app.is_key_pressed(KEY_A): player_obj['x'] -= PLAYER_SPEED
    if app.is_key_pressed(KEY_D): player_obj['x'] += PLAYER_SPEED
    app.moveto(player_id, player_obj['x'], player_obj['y'])
    app.rotate(player_id, 1)

    # 背景オブジェクトの移動
    for obj_data in objects:
        obj = app._draw_objects.get(obj_data['id'])
        if not obj: continue
        new_x = obj['x'] + obj_data['vx']
        new_y = obj['y'] + obj_data['vy']
        
        radius = obj.get('radius', 0)
        if new_x - radius < 0 or new_x + radius > WIDTH: obj_data['vx'] *= -1
        if new_y - radius < 0 or new_y + radius > HEIGHT: obj_data['vy'] *= -1
        app.moveto(obj_data['id'], new_x, new_y)

    # マウス追従オブジェクト
    mouse_x, mouse_y = app.get_mouse_position()
    app.moveto(player_ellipse_id, mouse_x, mouse_y)
    app.rotate(player_ellipse_id, -2)

    # 星の回転
    app.set_rotation(star_id, (time.time() * 50) % 360)
    app.moveto(star_id, WIDTH / 2, 100)

    # 線の更新
    app.delete_object(line_id)
    thickness = 5 + (math.sin(time.time() * 2) + 1) * 5
    line_id = app.draw_line(WIDTH / 2, HEIGHT, mouse_x, mouse_y, thickness=thickness, color=(100, 150, 255), z=11)

    # マウスクリック処理
    if app.is_mouse_button_pressed(MOUSE_BUTTON_LEFT):
        if click_circle_id != -1: app.delete_object(click_circle_id)
        if click_text_id != -1: app.delete_object(click_text_id)
        click_circle_id = app.draw_circle(mouse_x, mouse_y, 20, color=(255, 255, 0), z=10)
        click_text_id = app.draw_text(f"({int(mouse_x)}, {int(mouse_y)})", mouse_x + 25, mouse_y - 10, font="arial", size=16, color=(255, 255, 255), z=10)
    else:
        if click_circle_id != -1: app.delete_object(click_circle_id); click_circle_id = -1
        if click_text_id != -1: app.delete_object(click_text_id); click_text_id = -1

def game_draw():
    pass

# --- アプリケーションのセットアップと実行 ---
frame_count = 0
app.add_scene("title", setup=title_setup, update=title_update, draw=title_draw)
app.add_scene("game", setup=game_setup, update=game_update, draw=game_draw)

app.set_scene("title")
app.run()
