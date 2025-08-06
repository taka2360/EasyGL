import random
import time
import multiprocessing
from easy_gl import EasyGL, KEY_UP, KEY_DOWN

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720
INITIAL_OBJECTS = 30000
OBJECT_INCREMENT = 1000

# --- Object Data Structure ---
def create_object():
    """Creates a new object as a tuple for performance."""
    shape_type = random.randint(0, 1)  # 0 for circle, 1 for rectangle
    size = random.randint(10, 25) if shape_type == 0 else random.randint(15, 35)
    return (
        random.randint(50, WIDTH - 50),
        random.randint(50, HEIGHT - 50),
        random.uniform(-3, 3),
        random.uniform(-3, 3),
        size,
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
        shape_type,
        random.choice([0, 1, 2])
    )

# --- Worker Function for Multiprocessing ---
def update_worker(obj):
    """Update logic for a single object, designed to be run in a separate process."""
    x, y, vx, vy, size, r, g, b, shape_type, z = obj
    x += vx
    y += vy
    if x - size < 0 or x + size > WIDTH:
        vx *= -1
    if y - size < 0 or y + size > HEIGHT:
        vy *= -1
    return (x, y, vx, vy, size, r, g, b, shape_type, z)

# --- Main Application Logic (defined globally) ---
# These variables will be managed by the main process only.
objects = []
app = None 
fps, frame_count, last_fps_time = 0, 0, 0.0
key_up_pressed, key_down_pressed = False, False

def update(process_pool):
    """Core update loop, receives the process pool as an argument."""
    global objects, frame_count, last_fps_time, fps, key_up_pressed, key_down_pressed

    current_time = time.time()
    frame_count += 1
    if current_time - last_fps_time >= 1.0:
        fps = frame_count
        frame_count = 0
        last_fps_time = current_time

    if objects:
        objects = process_pool.map(update_worker, objects)

    if app.is_key_pressed(KEY_UP) and not key_up_pressed:
        objects.extend([create_object() for _ in range(OBJECT_INCREMENT)])
        key_up_pressed = True
    elif not app.is_key_pressed(KEY_UP):
        key_up_pressed = False

    if app.is_key_pressed(KEY_DOWN) and not key_down_pressed:
        objects = objects[:-OBJECT_INCREMENT] if len(objects) > OBJECT_INCREMENT else []
        key_down_pressed = True
    elif not app.is_key_pressed(KEY_DOWN):
        key_down_pressed = False

def draw():
    """Draws all objects and the HUD."""
    for obj in objects:
        x, y, _, _, size, r, g, b, shape_type, z = obj
        color = (r, g, b)
        if shape_type == 0:  # Circle
            app.draw_circle(x, y, size, color, z=z)
        else:  # Rectangle
            app.draw_rectangle(x, y, size, size, color, z=z)

    HUD_Z_ORDER = 100
    hud_color = (255, 255, 255)
    font = "arial"
    app.draw_text(f"FPS: {fps}", 10, 10, font, 24, hud_color, z=HUD_Z_ORDER)
    app.draw_text(f"Objects: {len(objects)}", 10, 40, font, 24, hud_color, z=HUD_Z_ORDER)
    app.draw_text(f"Processes: {multiprocessing.cpu_count()}", 10, 70, font, 24, hud_color, z=HUD_Z_ORDER)
    app.draw_text("UP Arrow: Add Objects", 10, HEIGHT - 60, font, 18, hud_color, z=HUD_Z_ORDER)
    app.draw_text("DOWN Arrow: Remove Objects", 10, HEIGHT - 35, font, 18, hud_color, z=HUD_Z_ORDER)

# --- Main Execution Guard ---
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # --- Initialization for the MAIN PROCESS ONLY ---
    app = EasyGL(title="Multiprocessing Stress Test", width=WIDTH, height=HEIGHT, max_fps=0)
    app.set_background_color(10, 20, 30)
    
    objects = [create_object() for _ in range(INITIAL_OBJECTS)]
    last_fps_time = time.time()

    process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    app.set_update_function(lambda: update(process_pool))
    app.set_draw_function(draw)
    
    app.run()

    # --- Cleanup ---
    print("Closing process pool...")
    process_pool.close()
    process_pool.join()
    print("Done.")