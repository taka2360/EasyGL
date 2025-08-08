import glfw
from OpenGL.GL import *
import math
from PIL import Image, ImageDraw, ImageFont
import time
import os
import platform
import numpy as np
from collections import defaultdict
from itertools import groupby
from numba import njit, types, float32, int32

# --- Numba-optimized Geometry Functions ---

@njit(float32(float32[:], float32[:], float32[:]), cache=True)
def _cross_product_2d_njit(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

@njit(types.boolean(float32[:], float32[:], float32[:], float32[:]), cache=True)
def _is_point_in_triangle_njit(pt, v1, v2, v3):
    d1 = _cross_product_2d_njit(v1, v2, pt)
    d2 = _cross_product_2d_njit(v2, v3, pt)
    d3 = _cross_product_2d_njit(v3, v1, pt)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

@njit(float32[:,:](float32[:,:], float32[:]), cache=True)
def _ear_clipping_njit(points, color_tuple):
    n = len(points)
    if n < 3:
        return np.empty((0, 5), dtype=np.float32)

    area = 0.0
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        area += (p1[0] * p2[1]) - (p2[0] * p1[1])
    
    local_points = points.copy()
    if area < 0:
        local_points = local_points[::-1]

    indices = np.arange(n, dtype=np.int32)
    triangles_verts = np.empty(((n - 2) * 3, 5), dtype=np.float32)
    tri_idx = 0
    num_points = n
    failsafe = 0
    r, g, b = color_tuple[0]/255.0, color_tuple[1]/255.0, color_tuple[2]/255.0

    while num_points > 2 and failsafe < n * 2:
        found_ear = False
        for i in range(num_points):
            prev_idx = indices[(i - 1 + num_points) % num_points]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % num_points]

            p_prev = local_points[prev_idx]
            p_curr = local_points[curr_idx]
            p_next = local_points[next_idx]

            if _cross_product_2d_njit(p_prev, p_curr, p_next) >= 0:
                is_ear = True
                for j_idx_ptr in range(num_points):
                    test_idx = indices[j_idx_ptr]
                    if test_idx != prev_idx and test_idx != curr_idx and test_idx != next_idx:
                        if _is_point_in_triangle_njit(local_points[test_idx], p_prev, p_curr, p_next):
                            is_ear = False
                            break
                
                if is_ear:
                    triangles_verts[tri_idx, 0] = p_prev[0]; triangles_verts[tri_idx, 1] = p_prev[1]
                    triangles_verts[tri_idx, 2] = r; triangles_verts[tri_idx, 3] = g; triangles_verts[tri_idx, 4] = b
                    tri_idx += 1
                    
                    triangles_verts[tri_idx, 0] = p_curr[0]; triangles_verts[tri_idx, 1] = p_curr[1]
                    triangles_verts[tri_idx, 2] = r; triangles_verts[tri_idx, 3] = g; triangles_verts[tri_idx, 4] = b
                    tri_idx += 1

                    triangles_verts[tri_idx, 0] = p_next[0]; triangles_verts[tri_idx, 1] = p_next[1]
                    triangles_verts[tri_idx, 2] = r; triangles_verts[tri_idx, 3] = g; triangles_verts[tri_idx, 4] = b
                    tri_idx += 1
                    
                    for k in range(i, num_points - 1):
                        indices[k] = indices[k + 1]
                    
                    num_points -= 1
                    found_ear = True
                    break
        
        if not found_ear:
            failsafe += 1
    return triangles_verts[:tri_idx]

@njit(float32[:,:](float32[:,:], float32, float32, float32), cache=True)
def _get_rotated_points_njit(points, center_x, center_y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rotated = np.empty_like(points)
    for i in range(len(points)):
        x, y = points[i]
        translated_x, translated_y = x - center_x, y - center_y
        rotated_x = translated_x * cos_a - translated_y * sin_a
        rotated_y = translated_x * sin_a + translated_y * cos_a
        rotated[i, 0] = rotated_x + center_x
        rotated[i, 1] = rotated_y + center_y
    return rotated

@njit(float32[:,:](int32, float32, float32, float32, float32, float32, float32[:]), cache=True)
def _transform_and_triangulate_njit(obj_type, x, y, w, h, angle, color):
    # obj_type: 0=rect, 1=ellipse/circle
    if obj_type == 0: # Rectangle
        half_w, half_h = w / 2, h / 2
        points = np.array([
            [x - half_w, y - half_h], [x + half_w, y - half_h],
            [x + half_w, y + half_h], [x - half_w, y + half_h]
        ], dtype=np.float32)
    elif obj_type == 1: # Ellipse/Circle
        points = np.empty((36, 2), dtype=np.float32)
        for i in range(36):
            a = math.radians(i * 10)
            points[i, 0] = x + w * math.cos(a)
            points[i, 1] = y + h * math.sin(a)
    else:
        return np.empty((0, 5), dtype=np.float32)

    if angle != 0:
        points = _get_rotated_points_njit(points, x, y, angle)
    
    return _ear_clipping_njit(points, color)

# --- シェーダー定義 ---
INSTANCE_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos; layout (location = 1) in vec2 aOffset; layout (location = 2) in vec2 aSize; layout (location = 3) in vec3 aColor; out vec3 ourColor; uniform mat4 projection; void main() { gl_Position = projection * vec4((aPos * aSize) + aOffset, 0.0, 1.0); ourColor = aColor; }"""
COLOR_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor; in vec3 ourColor; void main() { FragColor = vec4(ourColor, 1.0); }"""
TEXTURE_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos; layout (location = 1) in vec2 aTexCoord; out vec2 TexCoord; uniform mat4 projection; void main() { gl_Position = projection * vec4(aPos.x, aPos.y, 0.0, 1.0); TexCoord = aTexCoord; }"""
TEXTURE_FRAGMENT_SHADER = """#version 330 core
out vec4 FragColor; in vec2 TexCoord; uniform sampler2D ourTexture; void main() { FragColor = texture(ourTexture, TexCoord); }"""
SIMPLE_VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec2 aPos; layout (location = 1) in vec3 aColor; out vec3 ourColor; uniform mat4 projection; void main() { gl_Position = projection * vec4(aPos, 0.0, 1.0); ourColor = aColor; }"""

class EasyGL:
    MAX_INSTANCES = 20000
    MAX_TEXTURE_QUADS = 5000
    MAX_POLYGON_VERTICES = 800000

    def __init__(self, title="EasyGL", width=800, height=600, max_fps=60):
        if not glfw.init(): raise Exception("glfw can not be initialized!")
        self._width, self._height = width, height
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window: glfw.terminate(); raise Exception("glfw window can not be created!")
        glfw.make_context_current(self._window)
        glfw.set_window_pos(self._window, 400, 200)

        self._instance_shader = self._compile_shader(INSTANCE_VERTEX_SHADER, COLOR_FRAGMENT_SHADER)
        self._texture_shader = self._compile_shader(TEXTURE_VERTEX_SHADER, TEXTURE_FRAGMENT_SHADER)
        self._simple_shader = self._compile_shader(SIMPLE_VERTEX_SHADER, COLOR_FRAGMENT_SHADER)
        
        self._proj_mat_loc_instance = glGetUniformLocation(self._instance_shader, "projection")
        self._proj_mat_loc_texture = glGetUniformLocation(self._texture_shader, "projection")
        self._proj_mat_loc_simple = glGetUniformLocation(self._simple_shader, "projection")

        self._instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.MAX_INSTANCES * 7 * 4, None, GL_DYNAMIC_DRAW)

        self._rect_model_vbo, self._rect_vertex_count, self._rectangle_vao = self._create_shape_model([-0.5,-0.5, 0.5,-0.5, 0.5,0.5, -0.5,-0.5, 0.5,0.5, -0.5,0.5])
        self._circle_model_vbo, self._circle_vertex_count, self._circle_vao = self._create_shape_model(self._generate_circle_verts())

        self._texture_vao, self._texture_vbo = self._create_texture_vao()
        glBindBuffer(GL_ARRAY_BUFFER, self._texture_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.MAX_TEXTURE_QUADS * 6 * 4 * 4, None, GL_DYNAMIC_DRAW)

        self._polygon_vao, self._polygon_vbo = self._create_polygon_vao()
        glBindBuffer(GL_ARRAY_BUFFER, self._polygon_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.MAX_POLYGON_VERTICES * 5 * 4, None, GL_DYNAMIC_DRAW)

        self._draw_calls = defaultdict(lambda: {'rectangles': [], 'circles': [], 'textures': [], 'polygons': []})
        self._draw_objects = {}
        self._next_id = 0
        self._max_fps, self._frame_time, self._last_frame_time = max_fps, (1.0 / max_fps if max_fps > 0 else 0), 0
        self._keys, self._mouse_buttons, self._mouse_pos = {}, {}, (0, 0)
        self._background_color = (0.1, 0.2, 0.3, 1.0)
        
        glfw.set_key_callback(self._window, self._key_callback)
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
        
        self._scenes = {}
        self._current_scene = None
        self._active_update = None
        self._active_draw = None
        self._draw_function, self._update_function = None, None
        self._delta_time = 0.0

        self._textures, self._text_cache, self._font_path_cache, self._font_cache = {}, {}, {}, {}
        self._texture_dims = {}
        
        # Pre-compile Numba functions
        _transform_and_triangulate_njit(0, 0, 0, 0, 0, 0, np.array([0,0,0], dtype=np.float32))


    def _get_next_id(self):
        self._next_id += 1
        return self._next_id

    def moveto(self, object_id, x, y):
        if object_id in self._draw_objects:
            obj = self._draw_objects[object_id]
            dx = x - obj.get('x', x)
            dy = y - obj.get('y', y)
            if obj['type'] == 'polygon' and (dx != 0 or dy != 0):
                obj['points'] = [(px + dx, py + dy) for px, py in obj['points']]
            obj['x'], obj['y'] = x, y

    def rotate(self, object_id, angle_delta):
        if object_id in self._draw_objects:
            self._draw_objects[object_id]['angle'] = (self._draw_objects[object_id].get('angle', 0) + angle_delta) % 360

    def set_rotation(self, object_id, angle):
        if object_id in self._draw_objects:
            self._draw_objects[object_id]['angle'] = angle % 360

    def delete_object(self, object_id):
        if object_id in self._draw_objects:
            del self._draw_objects[object_id]

    def add_scene(self, name, setup, update, draw):
        self._scenes[name] = {'setup': setup, 'update': update, 'draw': draw}

    def set_scene(self, name):
        if name in self._scenes:
            self._current_scene = name
            scene = self._scenes[name]
            self._active_update, self._active_draw = scene['update'], scene['draw']
            if scene['setup']: scene['setup']()
        else:
            print(f"Warning: Scene '{name}' not found.")

    def _generate_circle_verts(self, segments=36):
        verts = []
        for i in range(segments):
            a1, a2 = (i/segments)*2*math.pi, ((i+1)/segments)*2*math.pi
            verts.extend([0.0,0.0, math.cos(a1)*0.5,math.sin(a1)*0.5, math.cos(a2)*0.5,math.sin(a2)*0.5])
        return verts

    def _create_shape_model(self, vertices):
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32).nbytes, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)
        vao = self._create_instanced_vao(vbo, self._instance_vbo)
        return vbo, len(vertices) // 2, vao

    def _create_instanced_vao(self, model_vbo, instance_vbo):
        vao = glGenVertexArrays(1); glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, model_vbo); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None); glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo); stride = 7 * 4
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(1); glVertexAttribDivisor(1, 1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8)); glEnableVertexAttribArray(2); glVertexAttribDivisor(2, 1)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16)); glEnableVertexAttribArray(3); glVertexAttribDivisor(3, 1)
        glBindVertexArray(0); return vao

    def _create_texture_vao(self):
        vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8)); glEnableVertexAttribArray(1)
        glBindVertexArray(0); return vao, vbo

    def _create_polygon_vao(self):
        vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); stride = 5 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8)); glEnableVertexAttribArray(1)
        glBindVertexArray(0); return vao, vbo

    def run(self):
        if self._update_function and self._draw_function and not self._scenes:
            self.add_scene("default", None, self._update_function, self._draw_function)
            self.set_scene("default")
        elif self._scenes and not self._current_scene:
            self.set_scene(next(iter(self._scenes)))

        self._last_frame_time = glfw.get_time()
        projection_matrix = np.array([[2/self._width,0,0,-1], [0,-2/self._height,0,1], [0,0,-1,0], [0,0,0,1]], dtype=np.float32).T
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        while not glfw.window_should_close(self._window):
            current_time = glfw.get_time()
            self._delta_time = current_time - self._last_frame_time
            if self._frame_time > 0 and self._delta_time < self._frame_time:
                time.sleep(self._frame_time - self._delta_time)
                current_time = glfw.get_time()
                self._delta_time = current_time - self._last_frame_time
            self._last_frame_time = current_time

            glClearColor(*self._background_color); glClear(GL_COLOR_BUFFER_BIT)

            if self._active_update: self._active_update()
            if self._active_draw: self._active_draw()
            
            self._prepare_draw_calls_from_objects()
            self._flush(projection_matrix)

            glfw.swap_buffers(self._window); glfw.poll_events()
        glfw.terminate()

    def get_delta_time(self):
        return self._delta_time

    def _prepare_draw_calls_from_objects(self):
        self._draw_calls.clear()
        color_np = np.empty(3, dtype=np.float32)

        for obj_id, obj in self._draw_objects.items():
            z = obj['z']
            obj_type = obj['type']
            angle = obj.get('angle', 0)

            if angle == 0 and obj_type in ['rectangle', 'circle', 'ellipse']:
                if obj_type == 'rectangle':
                    r, g, b = obj['color']
                    self._draw_calls[z]['rectangles'].append((obj['x'], obj['y'], obj['width'], obj['height'], r/255, g/255, b/255))
                else: # circle, ellipse
                    r, g, b = obj['color']
                    w = obj.get('radius', obj['width']) * (1 if obj_type == 'circle' else 1)
                    h = obj.get('radius', obj['height']) * (1 if obj_type == 'circle' else 1)
                    self._draw_calls[z]['circles'].append((obj['x'], obj['y'], w*2, h*2, r/255, g/255, b/255))
                continue

            if obj_type in ['rectangle', 'circle', 'ellipse']:
                color_np[0], color_np[1], color_np[2] = obj['color']
                if obj_type == 'rectangle':
                    triangles = _transform_and_triangulate_njit(0, obj['x'], obj['y'], obj['width'], obj['height'], angle, color_np)
                else: # circle, ellipse
                    w = obj.get('radius', obj['width'] / 2)
                    h = obj.get('radius', obj['height'] / 2)
                    triangles = _transform_and_triangulate_njit(1, obj['x'], obj['y'], w, h, angle, color_np)
                if triangles.size > 0:
                    self._draw_calls[z]['polygons'].append(triangles)

            elif obj_type == 'polygon':
                if len(obj['points']) < 3: continue
                color_np[0], color_np[1], color_np[2] = obj['color']
                points_np = np.array(obj['points'], dtype=np.float32)
                if angle != 0:
                    points_np = _get_rotated_points_njit(points_np, obj['x'], obj['y'], angle)
                triangles_verts = _ear_clipping_njit(points_np, color_np)
                if triangles_verts.size > 0:
                    self._draw_calls[z]['polygons'].append(triangles_verts)

            elif obj_type in ['image', 'text']:
                if obj_type == 'image':
                    tex_id, w, h = obj['tex_id'], obj['width'], obj['height']
                    if w == 'auto' or h == 'auto':
                        orig_w, orig_h = self._texture_dims.get(tex_id, (0,0))
                        if orig_w > 0 and orig_h > 0:
                            if w == 'auto' and h == 'auto': w, h = orig_w, orig_h
                            elif w == 'auto': w = h * (orig_w / orig_h)
                            else: h = w * (orig_h / orig_w)
                        else: continue
                else: # text
                    tex_id, w, h = self._create_text_texture(obj['text'], obj['font'], obj['size'], obj['color'])

                if tex_id != -1:
                    x, y = obj['x'], obj['y']
                    align_center = obj.get('align_center', False) if obj_type == 'image' else False
                    x1, y1 = (x - w / 2, y - h / 2) if align_center else (x, y)
                    x2, y2 = x1 + w, y1 + h
                    
                    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                    if angle != 0:
                        points = _get_rotated_points_njit(points, x, y, angle)
                    
                    (x1,y1),(x2,y1),(x2,y2),(x1,y2) = points
                    group_vertices = [x1,y1,0,0, x2,y1,1,0, x1,y2,0,1, x2,y1,1,0, x2,y2,1,1, x1,y2,0,1]
                    self._draw_calls[z]['textures'].append((tex_id, np.array(group_vertices, dtype=np.float32)))

    def _flush(self, projection_matrix):
        sorted_z = sorted(self._draw_calls.keys())
        for z in sorted_z:
            layer = self._draw_calls[z]
            self._flush_instanced_draws(projection_matrix, layer['rectangles'], layer['circles'])
            self._flush_polygons(projection_matrix, layer['polygons'])
            self._flush_textures_batched(projection_matrix, layer['textures'])

    def _flush_instanced_draws(self, projection_matrix, rectangles, circles):
        if not rectangles and not circles: return
        glUseProgram(self._instance_shader)
        glUniformMatrix4fv(self._proj_mat_loc_instance, 1, GL_FALSE, projection_matrix)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        
        if rectangles:
            data = np.array(rectangles, dtype=np.float32)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            glBindVertexArray(self._rectangle_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._rect_vertex_count, len(rectangles))
        
        if circles:
            data = np.array(circles, dtype=np.float32)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            glBindVertexArray(self._circle_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._circle_vertex_count, len(circles))
            
        glBindVertexArray(0); glUseProgram(0)

    def _flush_polygons(self, projection_matrix, polygons):
        if not polygons: return
        glUseProgram(self._simple_shader)
        glUniformMatrix4fv(self._proj_mat_loc_simple, 1, GL_FALSE, projection_matrix)
        glBindVertexArray(self._polygon_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._polygon_vbo)
        
        all_vertices = np.concatenate(polygons)
        glBufferSubData(GL_ARRAY_BUFFER, 0, all_vertices.nbytes, all_vertices)
        glDrawArrays(GL_TRIANGLES, 0, len(all_vertices))
        glBindVertexArray(0); glUseProgram(0)

    def _flush_textures_batched(self, projection_matrix, textures):
        if not textures: return
        glUseProgram(self._texture_shader)
        glUniformMatrix4fv(self._proj_mat_loc_texture, 1, GL_FALSE, projection_matrix)
        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self._texture_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._texture_vbo)

        textures.sort(key=lambda t: t[0])
        
        for tex_id, group in groupby(textures, key=lambda t: t[0]):
            all_vertices = np.concatenate([g[1] for g in group])
            if all_vertices.size == 0: continue
            
            glBufferSubData(GL_ARRAY_BUFFER, 0, all_vertices.nbytes, all_vertices)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glDrawArrays(GL_TRIANGLES, 0, len(all_vertices) // 4)

        glBindVertexArray(0); glUseProgram(0)
        
    def draw_rectangle(self, x, y, width, height, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {'type': 'rectangle', 'x': x, 'y': y, 'width': width, 'height': height, 'color': color, 'z': z, 'angle': 0}
        return obj_id

    def draw_circle(self, x, y, radius, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {'type': 'circle', 'x': x, 'y': y, 'radius': radius, 'color': color, 'z': z, 'angle': 0}
        return obj_id

    def draw_ellipse(self, x, y, width, height, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {'type': 'ellipse', 'x': x, 'y': y, 'width': width, 'height': height, 'color': color, 'z': z, 'angle': 0}
        return obj_id

    def draw_line(self, x1, y1, x2, y2, thickness=1, color=(255,255,255), z=0):
        if thickness <= 0: return -1
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0: return -1
        nx, ny = -dy / length, dx / length
        half_t = thickness / 2.0
        points = [(x1 - nx * half_t, y1 - ny * half_t), (x2 - nx * half_t, y2 - ny * half_t), (x2 + nx * half_t, y2 + ny * half_t), (x1 + nx * half_t, y1 + ny * half_t)]
        return self.draw_polygon(points, color, z)

    def draw_polygon(self, points, color=(255,255,255), z=0):
        if len(points) < 3: return -1
        obj_id = self._get_next_id()
        x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
        self._draw_objects[obj_id] = {'type': 'polygon', 'points': points, 'color': color, 'z': z, 'x': sum(x_coords) / len(points), 'y': sum(y_coords) / len(points), 'angle': 0}
        return obj_id

    def draw_text(self, text, x, y, font, size, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {'type': 'text', 'text': text, 'x': x, 'y': y, 'font': font, 'size': size, 'color': color, 'z': z, 'angle': 0}
        return obj_id

    def draw_image(self, tex_id, width, height, x, y, align_center=True, z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {'type': 'image', 'tex_id': tex_id, 'width': width, 'height': height, 'x': x, 'y': y, 'align_center': align_center, 'z': z, 'angle': 0}
        return obj_id

    def load_texture(self, filepath):
        return self._textures.get(filepath) or self._load_texture_file(filepath)

    def _compile_shader(self, vs, fs):
        p = glCreateProgram()
        s1, s2 = glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(s1, vs); glCompileShader(s1)
        glShaderSource(s2, fs); glCompileShader(s2)
        glAttachShader(p, s1); glAttachShader(p, s2)
        glLinkProgram(p); glDeleteShader(s1); glDeleteShader(s2)
        return p

    def set_background_color(self, r, g, b):
        self._background_color = (r / 255, g / 255, b / 255, 1.0)

    def _key_callback(self, w, k, sc, a, m): self._keys[k] = a != glfw.RELEASE
    def _mouse_button_callback(self, w, b, a, m): self._mouse_buttons[b] = a != glfw.RELEASE
    def _cursor_pos_callback(self, w, x, y): self._mouse_pos = (x, y)
    def is_key_pressed(self, k): return self._keys.get(k, False)
    def get_mouse_position(self): return self._mouse_pos
    def is_mouse_button_pressed(self, b): return self._mouse_buttons.get(b, False)

    def _find_font_path(self, f):
        f_lower = f.lower().replace(" ", "")
        if f_lower in self._font_path_cache: return self._font_path_cache[f_lower]
        d = os.path.join(os.environ.get("SystemRoot", "C:\Windows"), "Fonts") if platform.system() == "Windows" else "/System/Library/Fonts/Supplemental"
        fp = os.path.join(d, f_lower + ".ttf")
        self._font_path_cache[f_lower] = fp if os.path.exists(fp) else None
        return self._font_path_cache[f_lower]

    def _get_font(self, f, s):
        k = (f.lower(), s)
        return self._font_cache.get(k) or self._load_font(f, s, k)

    def _load_font(self, f, s, k):
        fp = self._find_font_path(f)
        self._font_cache[k] = (ImageFont.truetype(fp, s) if fp else ImageFont.load_default())
        return self._font_cache[k]

    def _create_text_texture(self, t, f, s, c):
        font = self._get_font(f, s)
        k = (t, font.path if hasattr(font, "path") else "default", s, c)
        if k in self._text_cache: return self._text_cache[k]
        tid, w, h = self._render_text_texture(t, font, c)
        self._text_cache[k] = (tid, w, h)
        return tid, w, h

    def _render_text_texture(self, t, font, c):
        try:
            bbox = font.getbbox(t)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            i = Image.new("RGBA", (w, h))
            d = ImageDraw.Draw(i)
            d.text((-bbox[0], -bbox[1]), t, font=font, fill=c + (255,))
        except AttributeError:
            w, h = font.getsize(t)
            i = Image.new("RGBA", (w, h))
            d = ImageDraw.Draw(i)
            d.text((0, 0), t, font=font, fill=c + (255,))
        return self._create_texture_from_image(i)

    def _create_texture_from_image(self, i):
        d = i.tobytes()
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, i.width, i.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, d)
        self._texture_dims[tid] = (i.width, i.height)
        return tid, i.width, i.height

    def _load_texture_file(self, p):
        try:
            i = Image.open(p).convert("RGBA")
            tid, _, _ = self._create_texture_from_image(i)
            self._textures[p] = tid
            return tid
        except FileNotFoundError:
            print(f"Error: Texture file not found at {p}")
            return -1

# --- GLFWキー定数 ---
KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H, KEY_I, KEY_J, KEY_K, KEY_L, KEY_M, KEY_N, KEY_O, KEY_P, KEY_Q, KEY_R, KEY_S, KEY_T, KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z = [getattr(glfw, f'KEY_{c}') for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9 = [getattr(glfw, f'KEY_{c}') for c in "0123456789"]
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE, KEY_ENTER, KEY_ESCAPE, KEY_TAB, KEY_BACKSPACE = [getattr(glfw, f'KEY_{n}') for n in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', 'ENTER', 'ESCAPE', 'TAB', 'BACKSPACE']]
MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE = glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE
''

class EasyGL:
    """
    EasyGL: GLFWとOpenGLをラップし、2Dグラフィックスを簡単に描画するためのクラス。

    主な特徴:
    - オブジェクトベースAPI: `draw_rectangle`等で描画オブジェクトを作成し、IDで操作します。
    - 高パフォーマンス: 背後では、インスタンシングやバッチ処理により、大量のオブジェクトも
      効率的に描画します。
    - 多彩な描画機能: 図形、画像、テキスト、凹多角形など、様々な要素を扱えます。

    基本的な使い方:
    ```python
    app = EasyGL(title="My App", width=1280, height=720)

    # 描画オブジェクトをセットアップ
    my_circle = app.draw_circle(100, 100, 50, color=(255, 0, 0))

    def update():
        # 毎フレームの状態更新
        app.rotate(my_circle, 1) # 1度ずつ回転

    app.set_update_function(update)
    # 新しいモデルでは、draw関数は通常空になります
    app.set_draw_function(lambda: None)
    app.run()
    ```
    """
    MAX_INSTANCES = 20000
    MAX_TEXTURE_QUADS = 5000
    MAX_POLYGON_VERTICES = 800000

    def __init__(self, title="EasyGL", width=800, height=600, max_fps=60):
        """
        EasyGLのウィンドウとOpenGLコンテキストを初期化します。

        Args:
            title (str): ウィンドウのタイトル。
            width (int): ウィンドウの幅。
            height (int): ウィンドウの高さ。
            max_fps (int): 最大フレームレート, デフォルト値は60です. 0を指定すると無制限（VSync依存）になります。
        """
        
        _ear_clipping_njit(np.array([[0,0],[1,0],[0,1]], dtype=np.float32), np.array([0,0,0], dtype=np.float32))
        
        if not glfw.init(): raise Exception("glfw can not be initialized!")
        self._width, self._height = width, height
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window: glfw.terminate(); raise Exception("glfw window can not be created!")
        glfw.make_context_current(self._window)
        glfw.set_window_pos(self._window, 400, 200)

        self._instance_shader = self._compile_shader(INSTANCE_VERTEX_SHADER, COLOR_FRAGMENT_SHADER)
        self._texture_shader = self._compile_shader(TEXTURE_VERTEX_SHADER, TEXTURE_FRAGMENT_SHADER)
        self._simple_shader = self._compile_shader(SIMPLE_VERTEX_SHADER, COLOR_FRAGMENT_SHADER)
        
        self._proj_mat_loc_instance = glGetUniformLocation(self._instance_shader, "projection")
        self._proj_mat_loc_texture = glGetUniformLocation(self._texture_shader, "projection")
        self._proj_mat_loc_simple = glGetUniformLocation(self._simple_shader, "projection")

        self._instance_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.MAX_INSTANCES * 7 * 4, None, GL_DYNAMIC_DRAW)

        self._rect_model_vbo, self._rect_vertex_count, self._rectangle_vao = self._create_shape_model([-0.5,-0.5, 0.5,-0.5, 0.5,0.5, -0.5,-0.5, 0.5,0.5, -0.5,0.5])
        self._circle_model_vbo, self._circle_vertex_count, self._circle_vao = self._create_shape_model(self._generate_circle_verts())

        self._texture_vao, self._texture_vbo = self._create_texture_vao()
        glBindBuffer(GL_ARRAY_BUFFER, self._texture_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.MAX_TEXTURE_QUADS * 6 * 4 * 4, None, GL_DYNAMIC_DRAW)

        self._polygon_vao, self._polygon_vbo = self._create_polygon_vao()
        glBindBuffer(GL_ARRAY_BUFFER, self._polygon_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.MAX_POLYGON_VERTICES * 5 * 4, None, GL_DYNAMIC_DRAW)

        self._draw_calls = defaultdict(lambda: {'rectangles': [], 'circles': [], 'textures': [], 'polygons': []})
        self._draw_objects = {}
        self._next_id = 0
        self._max_fps, self._frame_time, self._last_frame_time = max_fps, (1.0 / max_fps if max_fps > 0 else 0), 0
        self._keys, self._mouse_buttons, self._mouse_pos = {}, {}, (0, 0)
        self._background_color = (0.1, 0.2, 0.3, 1.0)
        
        glfw.set_key_callback(self._window, self._key_callback)
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
        
        self._scenes = {}
        self._current_scene = None
        self._active_update = None
        self._active_draw = None
        self._draw_function, self._update_function = None, None # For backward compatibility
        self._delta_time = 0.0

        self._textures, self._text_cache, self._font_path_cache, self._font_cache = {}, {}, {}, {}
        self._texture_dims = {}

    def _get_next_id(self):
        self._next_id += 1
        return self._next_id

    def moveto(self, object_id, x, y):
        """
        指定したIDのオブジェクトを新しい座標に移動します。

        Args:
            object_id (int): 移動するオブジェクトのID。
            x (float): 新しいx座標。
            y (float): 新しいy座標。
        """
        if object_id in self._draw_objects:
            obj = self._draw_objects[object_id]
            
            dx = x - obj.get('x', x)
            dy = y - obj.get('y', y)

            if obj['type'] == 'polygon' and (dx != 0 or dy != 0):
                obj['points'] = [(px + dx, py + dy) for px, py in obj['points']]
            
            obj['x'] = x
            obj['y'] = y

    def rotate(self, object_id, angle_delta):
        """
        オブジェクトを現在の角度から指定した角度（度数法）だけ回転させます。

        Args:
            object_id (int): 回転させるオブジェクトのID。
            angle_delta (float): 回転させる角度（度数法）。
        """
        if object_id in self._draw_objects:
            self._draw_objects[object_id]['angle'] = (self._draw_objects[object_id].get('angle', 0) + angle_delta) % 360

    def set_rotation(self, object_id, angle):
        """
        オブジェクトの回転角度を絶対角度（度数法）に設定します。

        Args:
            object_id (int): 回転させるオブジェクトのID。
            angle (float): 設定する絶対角度（度数法）。
        """
        if object_id in self._draw_objects:
            self._draw_objects[object_id]['angle'] = angle % 360

    def delete_object(self, object_id):
        """
        指定したIDのオブジェクトを描画リストから削除します。

        Args:
            object_id (int): 削除するオブジェクトのID。
        """
        if object_id in self._draw_objects:
            del self._draw_objects[object_id]

    def add_scene(self, name, setup, update, draw):
        self._scenes[name] = {'setup': setup, 'update': update, 'draw': draw}

    def set_scene(self, name):
        if name in self._scenes:
            self._current_scene = name
            scene = self._scenes[name]
            self._active_update = scene['update']
            self._active_draw = scene['draw']
            if scene['setup']:
                scene['setup']()
        else:
            print(f"Warning: Scene '{name}' not found.")

    def _generate_circle_verts(self, segments=36):
        verts = []
        for i in range(segments):
            a1, a2 = (i/segments)*2*math.pi, ((i+1)/segments)*2*math.pi
            verts.extend([0.0,0.0, math.cos(a1)*0.5,math.sin(a1)*0.5, math.cos(a2)*0.5,math.sin(a2)*0.5])
        return verts

    def _create_shape_model(self, vertices):
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, np.array(vertices, dtype=np.float32).nbytes, np.array(vertices, dtype=np.float32), GL_STATIC_DRAW)
        vao = self._create_instanced_vao(vbo, self._instance_vbo)
        return vbo, len(vertices) // 2, vao

    def _create_instanced_vao(self, model_vbo, instance_vbo):
        vao = glGenVertexArrays(1); glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, model_vbo); glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None); glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, instance_vbo); stride = 7 * 4
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(1); glVertexAttribDivisor(1, 1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8)); glEnableVertexAttribArray(2); glVertexAttribDivisor(2, 1)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16)); glEnableVertexAttribArray(3); glVertexAttribDivisor(3, 1)
        glBindVertexArray(0); return vao

    def _create_texture_vao(self):
        vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8)); glEnableVertexAttribArray(1)
        glBindVertexArray(0); return vao, vbo

    def _create_polygon_vao(self):
        vao, vbo = glGenVertexArrays(1), glGenBuffers(1)
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo); stride = 5 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8)); glEnableVertexAttribArray(1)
        glBindVertexArray(0); return vao, vbo

    def run(self):
        if self._update_function and self._draw_function and not self._scenes:
            self.add_scene("default", None, self._update_function, self._draw_function)
            self.set_scene("default")
        elif self._scenes and not self._current_scene:
            self.set_scene(next(iter(self._scenes)))

        self._last_frame_time = glfw.get_time()
        projection_matrix = np.array([[2/self._width,0,0,-1], [0,-2/self._height,0,1], [0,0,-1,0], [0,0,0,1]], dtype=np.float32).T
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        while not glfw.window_should_close(self._window):
            current_time = glfw.get_time()
            self._delta_time = current_time - self._last_frame_time
            if self._frame_time > 0 and self._delta_time < self._frame_time:
                time.sleep(self._frame_time - self._delta_time)
                current_time = glfw.get_time()
                self._delta_time = current_time - self._last_frame_time
            self._last_frame_time = current_time

            glClearColor(*self._background_color); glClear(GL_COLOR_BUFFER_BIT)

            if self._active_update: self._active_update()
            if self._active_draw: self._active_draw()
            
            self._prepare_draw_calls_from_objects()
            self._flush(projection_matrix)

            glfw.swap_buffers(self._window); glfw.poll_events()
        glfw.terminate()

    def get_delta_time(self):
        return self._delta_time

    def _get_rotated_points(self, points, center_x, center_y, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rotated = []
        for x, y in points:
            translated_x, translated_y = x - center_x, y - center_y
            rotated_x = translated_x * cos_a - translated_y * sin_a
            rotated_y = translated_x * sin_a + translated_y * cos_a
            rotated.append((rotated_x + center_x, rotated_y + center_y))
        return rotated

    def _prepare_draw_calls_from_objects(self):
        self._draw_calls.clear()
        for obj_id, obj in self._draw_objects.items():
            z = obj['z']
            obj_type = obj['type']
            angle = obj.get('angle', 0)

            # If rotated, most shapes become polygons
            if angle != 0 and obj_type in ['rectangle', 'ellipse', 'circle', 'image', 'text']:
                x, y = obj['x'], obj['y']
                color = obj.get('color', (255,255,255))
                
                if obj_type == 'rectangle':
                    w, h = obj['width']/2, obj['height']/2
                    points = [(-w, -h), (w, -h), (w, h), (-w, h)]
                    points = [(p[0] + x, p[1] + y) for p in points]
                elif obj_type in ['circle', 'ellipse']:
                    if obj_type == 'circle':
                        w = h = obj['radius']
                    else: # ellipse
                        w = obj['width'] / 2
                        h = obj['height'] / 2
                    points = []
                    for i in range(36):
                        a = math.radians(i * 10)
                        points.append((x + w * math.cos(a), y + h * math.sin(a)))
                elif obj_type in ['image', 'text']:
                    if obj_type == 'image':
                        tex_id, w, h = obj['tex_id'], obj['width'], obj['height']
                        if w == 'auto' or h == 'auto': # Resolve auto dimensions
                            orig_w, orig_h = self._texture_dims.get(tex_id, (0,0))
                            if orig_w > 0 and orig_h > 0:
                                if w == 'auto' and h == 'auto': w, h = orig_w, orig_h
                                elif w == 'auto': w = h * (orig_w / orig_h)
                                else: h = w * (orig_h / orig_w)
                            else: continue
                    else: # text
                        tex_id, w, h = self._create_text_texture(obj['text'], obj['font'], obj['size'], obj['color'])

                    if tex_id != -1:
                        align_center = obj.get('align_center', False) if obj_type == 'image' else False
                        x1, y1 = (x - w / 2, y - h / 2) if align_center else (x, y)
                        x2, y2 = x1 + w, y1 + h
                        
                        # Rotate texture vertices
                        center_x, center_y = x, y
                        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        rotated_points = self._get_rotated_points(points, center_x, center_y, angle)
                        
                        # Add to texture draw calls with rotated vertices
                        (x1,y1),(x2,y1),(x2,y2),(x1,y2) = rotated_points
                        group_vertices = [x1,y1,0,0, x2,y1,1,0, x1,y2,0,1, x2,y1,1,0, x2,y2,1,1, x1,y2,0,1]
                        self._draw_calls[z]['textures'].append((tex_id, np.array(group_vertices, dtype=np.float32)))
                    continue # Skip to next object

                # Convert to polygon and add to draw calls
                rotated_points = self._get_rotated_points(points, x, y, angle)
                points_np = np.array(rotated_points, dtype=np.float32)
                color_np = np.array(color, dtype=np.float32)
                triangles_verts = _ear_clipping_njit(points_np, color_np)
                if triangles_verts.size > 0:
                    self._draw_calls[z]['polygons'].append(triangles_verts)
                continue

            # Default (non-rotated) drawing logic
            if obj_type == 'rectangle':
                r, g, b = obj['color']
                self._draw_calls[z]['rectangles'].append((obj['x'], obj['y'], obj['width'], obj['height'], r/255, g/255, b/255))
            elif obj_type == 'circle':
                r, g, b = obj['color']
                radius = obj['radius']
                self._draw_calls[z]['circles'].append((obj['x'], obj['y'], radius*2, radius*2, r/255, g/255, b/255))
            elif obj_type == 'ellipse':
                r, g, b = obj['color']
                self._draw_calls[z]['circles'].append((obj['x'], obj['y'], obj['width'], obj['height'], r/255, g/255, b/255))
            elif obj_type == 'polygon':
                if len(obj['points']) < 3: continue
                points_np = np.array(self._get_rotated_points(obj['points'], obj['x'], obj['y'], angle), dtype=np.float32)
                color_np = np.array(obj['color'], dtype=np.float32)
                triangles_verts = _ear_clipping_njit(points_np, color_np)
                if triangles_verts.size > 0:
                    self._draw_calls[z]['polygons'].append(triangles_verts)
            elif obj_type == 'text':
                tex_id, width, height = self._create_text_texture(obj['text'], obj['font'], obj['size'], obj['color'])
                if tex_id != -1:
                    x1, y1, x2, y2 = obj['x'], obj['y'], obj['x'] + width, obj['y'] + height
                    group_vertices = [x1,y1,0,0, x2,y1,1,0, x1,y2,0,1, x2,y1,1,0, x2,y2,1,1, x1,y2,0,1]
                    self._draw_calls[z]['textures'].append((tex_id, np.array(group_vertices, dtype=np.float32)))
            elif obj_type == 'image':
                tex_id = obj['tex_id']
                if tex_id == -1: continue
                draw_width, draw_height = obj['width'], obj['height']
                orig_dims = self._texture_dims.get(tex_id)
                if not orig_dims:
                    if draw_width == 'auto' or draw_height == 'auto': continue
                else:
                    if draw_width == 'auto' or draw_height == 'auto':
                        orig_width, orig_height = orig_dims
                        if orig_width == 0 or orig_height == 0: continue
                        if draw_width == 'auto' and draw_height == 'auto': draw_width, draw_height = orig_width, orig_height
                        elif draw_width == 'auto': draw_width = draw_height * (orig_width / orig_height)
                        elif draw_height == 'auto': draw_height = draw_width * (orig_height / orig_width)
                
                x, y, align_center = obj['x'], obj['y'], obj['align_center']
                x1, y1 = (x - draw_width / 2, y - draw_height / 2) if align_center else (x, y)
                x2, y2 = x1 + draw_width, y1 + draw_height
                group_vertices = [x1,y1,0,0, x2,y1,1,0, x1,y2,0,1, x2,y1,1,0, x2,y2,1,1, x1,y2,0,1]
                self._draw_calls[z]['textures'].append((tex_id, np.array(group_vertices, dtype=np.float32)))

    def _flush(self, projection_matrix):
        sorted_z = sorted(self._draw_calls.keys())
        for z in sorted_z:
            layer = self._draw_calls[z]
            self._flush_instanced_draws(projection_matrix, layer['rectangles'], layer['circles'])
            self._flush_polygons(projection_matrix, layer['polygons'])
            self._flush_textures_batched(projection_matrix, layer['textures'])

    def _flush_instanced_draws(self, projection_matrix, rectangles, circles):
        if not rectangles and not circles: return
        glUseProgram(self._instance_shader)
        glUniformMatrix4fv(self._proj_mat_loc_instance, 1, GL_FALSE, projection_matrix)
        glBindBuffer(GL_ARRAY_BUFFER, self._instance_vbo)
        
        if rectangles:
            data = np.array(rectangles, dtype=np.float32)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            glBindVertexArray(self._rectangle_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._rect_vertex_count, len(rectangles))
        
        if circles:
            data = np.array(circles, dtype=np.float32)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            glBindVertexArray(self._circle_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._circle_vertex_count, len(circles))
            
        glBindVertexArray(0); glUseProgram(0)

    def _flush_polygons(self, projection_matrix, polygons):
        if not polygons: return
        glUseProgram(self._simple_shader)
        glUniformMatrix4fv(self._proj_mat_loc_simple, 1, GL_FALSE, projection_matrix)
        glBindVertexArray(self._polygon_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._polygon_vbo)
        
        all_vertices = np.concatenate(polygons).astype(np.float32)
        glBufferSubData(GL_ARRAY_BUFFER, 0, all_vertices.nbytes, all_vertices)
        
        glDrawArrays(GL_TRIANGLES, 0, len(all_vertices))

        glBindVertexArray(0); glUseProgram(0)

    def _flush_textures_batched(self, projection_matrix, textures):
        if not textures: return
        glUseProgram(self._texture_shader)
        glUniformMatrix4fv(self._proj_mat_loc_texture, 1, GL_FALSE, projection_matrix)
        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self._texture_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._texture_vbo)

        textures.sort(key=lambda t: t[0])
        
        for tex_id, group in groupby(textures, key=lambda t: t[0]):
            all_vertices = np.concatenate([g[1] for g in group])
            if not all_vertices.any(): continue
            
            glBufferSubData(GL_ARRAY_BUFFER, 0, all_vertices.nbytes, all_vertices)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glDrawArrays(GL_TRIANGLES, 0, len(all_vertices) // 4)

        glBindVertexArray(0); glUseProgram(0)
        
    def draw_rectangle(self, x, y, width, height, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {
            'type': 'rectangle', 'x': x, 'y': y, 'width': width, 'height': height,
            'color': color, 'z': z, 'angle': 0
        }
        return obj_id

    def draw_circle(self, x, y, radius, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {
            'type': 'circle', 'x': x, 'y': y, 'radius': radius,
            'color': color, 'z': z, 'angle': 0
        }
        return obj_id

    def draw_ellipse(self, x, y, width, height, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {
            'type': 'ellipse', 'x': x, 'y': y, 'width': width, 'height': height,
            'color': color, 'z': z, 'angle': 0
        }
        return obj_id

    def draw_line(self, x1, y1, x2, y2, thickness=1, color=(255,255,255), z=0):
        if thickness <= 0: return -1
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0: return -1
        nx, ny = -dy / length, dx / length
        half_t = thickness / 2.0
        points = [
            (x1 - nx * half_t, y1 - ny * half_t),
            (x2 - nx * half_t, y2 - ny * half_t),
            (x2 + nx * half_t, y2 + ny * half_t),
            (x1 + nx * half_t, y1 + ny * half_t)
        ]
        return self.draw_polygon(points, color, z)

    def draw_polygon(self, points, color=(255,255,255), z=0):
        if len(points) < 3: return -1
        obj_id = self._get_next_id()
        x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
        self._draw_objects[obj_id] = {
            'type': 'polygon', 'points': points, 'color': color, 'z': z,
            'x': sum(x_coords) / len(points), 'y': sum(y_coords) / len(points), 'angle': 0
        }
        return obj_id

    def draw_text(self, text, x, y, font, size, color=(255,255,255), z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {
            'type': 'text', 'text': text, 'x': x, 'y': y, 
            'font': font, 'size': size, 'color': color, 'z': z, 'angle': 0
        }
        return obj_id

    def draw_image(self, tex_id, width, height, x, y, align_center=True, z=0):
        obj_id = self._get_next_id()
        self._draw_objects[obj_id] = {
            'type': 'image', 'tex_id': tex_id, 'width': width, 'height': height,
            'x': x, 'y': y, 'align_center': align_center, 'z': z, 'angle': 0
        }
        return obj_id

    def load_texture(self, filepath):
        return self._textures.get(filepath) or self._load_texture_file(filepath)

    def _compile_shader(self, vs, fs):
        p = glCreateProgram()
        s1, s2 = glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(s1, vs); glCompileShader(s1)
        glShaderSource(s2, fs); glCompileShader(s2)
        glAttachShader(p, s1); glAttachShader(p, s2)
        glLinkProgram(p); glDeleteShader(s1); glDeleteShader(s2)
        return p

    def set_background_color(self, r, g, b):
        self._background_color = (r / 255, g / 255, b / 255, 1.0)

    def _key_callback(self, w, k, sc, a, m): self._keys[k] = a != glfw.RELEASE
    def _mouse_button_callback(self, w, b, a, m): self._mouse_buttons[b] = a != glfw.RELEASE
    def _cursor_pos_callback(self, w, x, y): self._mouse_pos = (x, y)
    def is_key_pressed(self, k): return self._keys.get(k, False)
    def get_mouse_position(self): return self._mouse_pos
    def is_mouse_button_pressed(self, b): return self._mouse_buttons.get(b, False)

    def _find_font_path(self, f):
        f_lower = f.lower().replace(" ", "")
        if f_lower in self._font_path_cache: return self._font_path_cache[f_lower]
        d = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts") if platform.system() == "Windows" else "/System/Library/Fonts/Supplemental"
        fp = os.path.join(d, f_lower + ".ttf")
        self._font_path_cache[f_lower] = fp if os.path.exists(fp) else None
        return self._font_path_cache[f_lower]

    def _get_font(self, f, s):
        k = (f.lower(), s)
        return self._font_cache.get(k) or self._load_font(f, s, k)

    def _load_font(self, f, s, k):
        fp = self._find_font_path(f)
        self._font_cache[k] = (ImageFont.truetype(fp, s) if fp else ImageFont.load_default())
        return self._font_cache[k]

    def _create_text_texture(self, t, f, s, c):
        font = self._get_font(f, s)
        k = (t, font.path if hasattr(font, "path") else "default", s, c)
        if k in self._text_cache: return self._text_cache[k]
        tid, w, h = self._render_text_texture(t, font, c)
        self._text_cache[k] = (tid, w, h)
        return tid, w, h

    def _render_text_texture(self, t, font, c):
        try: bbox = font.getbbox(t); w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]; i = Image.new("RGBA", (w, h)); d = ImageDraw.Draw(i); d.text((-bbox[0], -bbox[1]), t, font=font, fill=c + (255,))
        except AttributeError: w, h = font.getsize(t); i = Image.new("RGBA", (w, h)); d = ImageDraw.Draw(i); d.text((0, 0), t, font=font, fill=c + (255,))
        return self._create_texture_from_image(i)

    def _create_texture_from_image(self, i):
        d = i.tobytes()
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, i.width, i.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, d)
        self._texture_dims[tid] = (i.width, i.height)
        return tid, i.width, i.height

    def _load_texture_file(self, p):
        try: i = Image.open(p).convert("RGBA"); tid, _, _ = self._create_texture_from_image(i); self._textures[p] = tid; return tid
        except FileNotFoundError: print(f"Error: Texture file not found at {p}"); return -1

# --- GLFWキー定数 ---
KEY_A, KEY_B, KEY_C, KEY_D, KEY_E, KEY_F, KEY_G, KEY_H, KEY_I, KEY_J, KEY_K, KEY_L, KEY_M, KEY_N, KEY_O, KEY_P, KEY_Q, KEY_R, KEY_S, KEY_T, KEY_U, KEY_V, KEY_W, KEY_X, KEY_Y, KEY_Z = [getattr(glfw, f'KEY_{c}') for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9 = [getattr(glfw, f'KEY_{c}') for c in "0123456789"]
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SPACE, KEY_ENTER, KEY_ESCAPE, KEY_TAB, KEY_BACKSPACE = [getattr(glfw, f'KEY_{n}') for n in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'SPACE', 'ENTER', 'ESCAPE', 'TAB', 'BACKSPACE']]
MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE = glfw.MOUSE_BUTTON_LEFT,glfw.MOUSE_BUTTON_RIGHT,glfw.MOUSE_BUTTON_MIDDLE
