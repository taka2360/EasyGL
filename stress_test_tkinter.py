import tkinter as tk
import random
import time
WIDTH, HEIGHT = 1280, 720
INITIAL_OBJECTS = 100
OBJECT_INCREMENT = 100

class MovingObject:
    def __init__(self):
        self.shape_type = random.choice(['circle', 'rectangle'])
        self.x = random.randint(50, WIDTH - 50)
        self.y = random.randint(50, HEIGHT - 50)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-150, 150) # Speed in pixels per second
        if self.shape_type == 'circle':
            self.size = random.randint(10, 25)
        else:
            self.size = random.randint(15, 35)
        self.color = "#%02x%02x%02x" % (
            random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.z = random.choice([0, 1, 2])  # Z-layer

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x - self.size < 0 or self.x + self.size > WIDTH: self.vx *= -1
        if self.y - self.size < 0 or self.y + self.size > HEIGHT: self.vy *= -1

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("tkinter Z-Order Stress Test")
        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg="#0a141e")
        self.canvas.pack()
        self.objects = [MovingObject() for _ in range(INITIAL_OBJECTS)]
        
        self.last_frame_time = time.time()
        self.last_fps_update_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps

        self.root.bind("<KeyPress-Up>", self.add_objects)
        self.root.bind("<KeyPress-Down>", self.remove_objects)

        self.update_loop()

        self.root.mainloop()

    def add_objects(self, event):
        for _ in range(OBJECT_INCREMENT):
            self.objects.append(MovingObject())

    def remove_objects(self, event):
        for _ in range(OBJECT_INCREMENT):
            if self.objects:
                self.objects.pop()

    def update_loop(self):
        start_time = time.time()
        dt = start_time - self.last_frame_time
        self.last_frame_time = start_time

        # Update logic
        for obj in self.objects:
            obj.update(dt)

        # Drawing logic
        self.canvas.delete("all")
        sorted_objects = sorted(self.objects, key=lambda o: o.z)
        for obj in sorted_objects:
            if obj.shape_type == 'circle':
                self.canvas.create_oval(
                    obj.x - obj.size, obj.y - obj.size,
                    obj.x + obj.size, obj.y + obj.size,
                    fill=obj.color, outline=""
                )
            else:
                self.canvas.create_rectangle(
                    obj.x - obj.size, obj.y - obj.size,
                    obj.x + obj.size, obj.y + obj.size,
                    fill=obj.color, outline=""
                )

        # --- HUD ---
        self.frame_count += 1
        if start_time - self.last_fps_update_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_update_time = start_time

        hud_color = "#ffffff"
        self.canvas.create_text(10, 10, anchor="nw", text=f"FPS: {self.fps}", fill=hud_color, font=("Arial", 16))
        self.canvas.create_text(10, 35, anchor="nw", text=f"Objects: {len(self.objects)}", fill=hud_color, font=("Arial", 16))
        self.canvas.create_text(10, HEIGHT - 60, anchor="nw", text="UP Arrow: Add Objects", fill=hud_color, font=("Arial", 14))
        self.canvas.create_text(10, HEIGHT - 35, anchor="nw", text="DOWN Arrow: Remove Objects", fill=hud_color, font=("Arial", 14))

        # Schedule next frame
        end_time = time.time()
        process_time = end_time - start_time
        wait_time = max(1, int((self.frame_time - process_time) * 1000))
        self.root.after(wait_time, self.update_loop)

App()
