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
from numba import njit

# --- Numba-optimized Ear Clipping ---
@njit
def _cross_product_2d_njit(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

@njit
def _is_point_in_triangle_njit(pt, v1, v2, v3):
    d1 = _cross_product_2d_njit(v1, v2, pt)
    d2 = _cross_product_2d_njit(v2, v3, pt)
    d3 = _cross_product_2d_njit(v3, v1, pt)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

@njit
def _ear_clipping_njit(points, color_tuple):
    n = len(points)
    if n < 3:
        return np.empty((0, 5), dtype=np.float32)

    # Ensure winding order is counter-clockwise
    area = 0.0
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        area += (p1[0] * p2[1]) - (p2[0] * p1[1])
    
    local_points = points.copy()
    if area < 0: # If area is negative, the winding is clockwise, so reverse to CCW.
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
                    triangles_verts[tri_idx, 0] = p_prev[0]
                    triangles_verts[tri_idx, 1] = p_prev[1]
                    triangles_verts[tri_idx, 2] = r; triangles_verts[tri_idx, 3] = g; triangles_verts[tri_idx, 4] = b;
                    tri_idx += 1
                    
                    triangles_verts[tri_idx, 0] = p_curr[0]
                    triangles_verts[tri_idx, 1] = p_curr[1]
                    triangles_verts[tri_idx, 2] = r; triangles_verts[tri_idx, 3] = g; triangles_verts[tri_idx, 4] = b;
                    tri_idx += 1

                    triangles_verts[tri_idx, 0] = p_next[0]
                    triangles_verts[tri_idx, 1] = p_next[1]
                    triangles_verts[tri_idx, 2] = r; triangles_verts[tri_idx, 3] = g; triangles_verts[tri_idx, 4] = b;
                    tri_idx += 1
                    
                    # Remove the ear tip index by shifting
                    for k in range(i, num_points - 1):
                        indices[k] = indices[k + 1]
                    
                    num_points -= 1
                    found_ear = True
                    break
        
        if not found_ear:
            failsafe += 1

    return triangles_verts[:tri_idx]

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
    """
    EasyGL: GLFWとOpenGLをラップし、2Dグラフィックスを簡単に描画するためのクラス。

    主な特徴:
    - 直感的なAPI: `draw_rectangle`, `draw_circle` など、分かりやすい関数で描画できます。
    - 高パフォーマンス: 背後では、インスタンシングやバッチ処理により、大量のオブジェクトも
      効率的に描画します。
    - 多彩な描画機能: 図形、画像、テキスト、凹多角形など、様々な要素を扱えます。

    基本的な使い方:
    ```python
    app = EasyGL(title="My App", width=1280, height=720)

    def update():
        # 毎フレームの状態更新
        pass

    def draw():
        # 描画処理
        app.draw_circle(400, 300, 50, color=(255, 0, 0))

    app.set_update_function(update)
    app.set_draw_function(draw)
    app.run()
    ```
    """
    MAX_INSTANCES = 20000
    MAX_TEXTURE_QUADS = 5000
    MAX_POLYGON_VERTICES = 30000

    def __init__(self, title="EasyGL", width=800, height=600, max_fps=60):
        """
        EasyGLのウィンドウとOpenGLコンテキストを初期化します。

        Args:
            title (str): ウィンドウのタイトル。
            width (int): ウィンドウの幅。
            height (int): ウィンドウの高さ。
            max_fps (int): 最大フレームレート。0を指定すると無制限（VSync依存）になります。
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
        self._max_fps, self._frame_time, self._last_frame_time = max_fps, (1.0 / max_fps if max_fps > 0 else 0), 0
        self._keys, self._mouse_buttons, self._mouse_pos = {}, {}, (0, 0)
        self._background_color = (0.1, 0.2, 0.3, 1.0)
        
        glfw.set_key_callback(self._window, self._key_callback)
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
        
        self._draw_function, self._update_function = None, None
        self._textures, self._text_cache, self._font_path_cache, self._font_cache = {}, {}, {}, {}
        self._texture_dims = {}



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
        """
        メインループを開始します。
        この関数を呼び出すと、ウィンドウが閉じるまで処理がブロックされます。
        `set_update_function` と `set_draw_function` で登録した関数が毎フレーム呼び出されます。
        """
        self._last_frame_time = glfw.get_time()
        projection_matrix = np.array([[2/self._width,0,0,-1], [0,-2/self._height,0,1], [0,0,-1,0], [0,0,0,1]], dtype=np.float32).T
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        while not glfw.window_should_close(self._window):
            current_time = glfw.get_time()
            delta_time = current_time - self._last_frame_time
            if self._frame_time > 0 and delta_time < self._frame_time:
                time.sleep(self._frame_time - delta_time)
            self._last_frame_time = glfw.get_time()

            glClearColor(*self._background_color); glClear(GL_COLOR_BUFFER_BIT)

            if self._update_function: self._update_function()
            self._draw_calls.clear()
            if self._draw_function: self._draw_function()
            
            self._flush(projection_matrix)

            glfw.swap_buffers(self._window); glfw.poll_events()
        glfw.terminate()

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
        
        vertex_count = len(all_vertices) // 5
        glDrawArrays(GL_TRIANGLES, 0, vertex_count)

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
            group_vertices = []
            for _, width, height, x, y, align_center in list(group):
                x1, y1 = (x - width / 2, y - height / 2) if align_center else (x, y)
                x2, y2 = x1 + width, y1 + height
                group_vertices.extend([x1,y1,0,0, x2,y1,1,0, x1,y2,0,1, x2,y1,1,0, x2,y2,1,1, x1,y2,0,1])

            if not group_vertices: continue
            data = np.array(group_vertices, dtype=np.float32)
            glBufferSubData(GL_ARRAY_BUFFER, 0, data.nbytes, data)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glDrawArrays(GL_TRIANGLES, 0, len(group_vertices) // 4)

        glBindVertexArray(0); glUseProgram(0)
        
    def draw_rectangle(self, x, y, width, height, color=(255,255,255), z=0):
        """
        矩形を描画リストに追加します。

        Args:
            x (float): 矩形の中心のx座標。
            y (float): 矩形の中心のy座標。
            width (float): 矩形の幅。
            height (float): 矩形の高さ。
            color (tuple): (R, G, B) のタプル (各0-255)。
            z (int): 描画の奥行き順。値が小さいほど奥に描画されます。
        """
        self._draw_calls[z]['rectangles'].append((x, y, width, height, color[0]/255, color[1]/255, color[2]/255))

    def draw_circle(self, x, y, radius, color=(255,255,255), z=0):
        """
        円を描画リストに追加します。

        Args:
            x (float): 円の中心のx座標。
            y (float): 円の中心のy座標。
            radius (float): 円の半径。
            color (tuple): (R, G, B) のタプル (各0-255)。
            z (int): 描画の奥行き順。
        """
        self._draw_calls[z]['circles'].append((x, y, radius*2, radius*2, color[0]/255, color[1]/255, color[2]/255))

    def draw_ellipse(self, x, y, width, height, color=(255,255,255), z=0):
        """
        楕円を描画リストに追加します。

        Args:
            x (float): 楕円の中心のx座標。
            y (float): 楕円の中心のy座標。
            width (float): 楕円の幅。
            height (float): 楕円の高さ。
            color (tuple): (R, G, B) のタプル (各0-255)。
            z (int): 描画の奥行き順。
        """
        self._draw_calls[z]['circles'].append((x, y, width, height, color[0]/255, color[1]/255, color[2]/255))

    def draw_polygon(self, points, color=(255,255,255), z=0):
        """
        多角形を描画リストに追加します。内部でEar Clipping法により三角形に分割されます。
        星形のような凹んだ多角形にも対応しています。

        Args:
            points (list of tuple): `[(x1, y1), (x2, y2), ...]` のような頂点のリスト。
            color (tuple): (R, G, B) のタプル (各0-255)。
            z (int): 描画の奥行き順。
        """
        if len(points) < 3: return

        points_np = np.array(points, dtype=np.float32)
        color_np = np.array(color, dtype=np.float32)

        # Call the JIT-compiled function
        triangles_verts = _ear_clipping_njit(points_np, color_np)

        if triangles_verts.size > 0:
            self._draw_calls[z]['polygons'].append(triangles_verts.flatten())

    def draw_text(self, text, x, y, font, size, color=(255,255,255), z=0):
        """
        テキストを描画リストに追加します。

        Args:
            text (str): 表示する文字列。
            x (float): テキストの左上のx座標。
            y (float): テキストの左上のy座標。
            font (str): `'arial'`, `'times new roman'` など、システムフォント名。
            size (int): フォントサイズ。
            color (tuple): (R, G, B) のタプル (各0-255)。
            z (int): 描画の奥行き順。
        """
        tex_id, width, height = self._create_text_texture(text, font, size, color)
        if tex_id != -1: self._draw_calls[z]['textures'].append((tex_id, width, height, x, y, False))

    def draw_image(self, tex_id, width, height, x, y, align_center=True, z=0):
        """
        画像を描画リストに追加します。

        Args:
            tex_id (int): `load_texture` で取得したテクスチャID。
            width (float or 'auto'): 描画する幅。'auto'でアスペクト比を維持します。
            height (float or 'auto'): 描画する高さ。'auto'でアスペクト比を維持します。
            x (float): 描画位置のx座標。
            y (float): 描画位置のy座標。
            align_center (bool): Trueの場合、(x,y)が画像の中心になります。
            z (int): 描画の奥行き順。
        """
        if tex_id == -1: return
        draw_width, draw_height = width, height
        orig_dims = self._texture_dims.get(tex_id)
        if not orig_dims:
            if width == 'auto' or height == 'auto': return
        else:
            if width == 'auto' or height == 'auto':
                orig_width, orig_height = orig_dims
                if orig_width == 0 or orig_height == 0: return
                if width == 'auto' and height == 'auto': draw_width, draw_height = orig_width, orig_height
                elif width == 'auto': draw_width = height * (orig_width / orig_height)
                elif height == 'auto': draw_height = width * (orig_height / orig_width)
        self._draw_calls[z]['textures'].append((tex_id, draw_width, draw_height, x, y, align_center))

    def load_texture(self, filepath):
        """
        画像ファイルを読み込み、テクスチャとして利用可能にします。

        Args:
            filepath (str): 画像ファイルのパス。

        Returns:
            int: 生成されたテクスチャのID。`draw_image`で使用します。
        """
        return self._textures.get(filepath) or self._load_texture_file(filepath)

    def _compile_shader(self, vs, fs): p=glCreateProgram();s1=glCreateShader(GL_VERTEX_SHADER);glShaderSource(s1,vs);glCompileShader(s1);p_s=glGetShaderiv(s1,GL_COMPILE_STATUS);s2=glCreateShader(GL_FRAGMENT_SHADER);glShaderSource(s2,fs);glCompileShader(s2);f_s=glGetShaderiv(s2,GL_COMPILE_STATUS);glAttachShader(p,s1);glAttachShader(p,s2);glLinkProgram(p);glDeleteShader(s1);glDeleteShader(s2);return p
    def set_draw_function(self, func): self._draw_function = func
    def set_update_function(self, func): self._update_function = func
    def set_background_color(self, r, g, b): self._background_color = (r/255, g/255, b/255, 1.0)
    def _key_callback(self,w,k,sc,a,m): self._keys[k] = a != glfw.RELEASE
    def _mouse_button_callback(self,w,b,a,m): self._mouse_buttons[b] = a != glfw.RELEASE
    def _cursor_pos_callback(self,w,x,y): self._mouse_pos = (x,y)
    def is_key_pressed(self,k): return self._keys.get(k, False)
    def get_mouse_position(self): return self._mouse_pos
    def is_mouse_button_pressed(self,b): return self._mouse_buttons.get(b, False)
    def _find_font_path(self,f):f=f.lower().replace(" ","");p=self._font_path_cache.get(f);return p if p else self._find_font_path_os(f)
    def _find_font_path_os(self,f):d=os.path.join(os.environ.get("SystemRoot","C:\\Windows"),"Fonts") if platform.system()=="Windows" else "/System/Library/Fonts/Supplemental";fp=os.path.join(d,f+".ttf");self._font_path_cache[f]=fp if os.path.exists(fp) else None;return self._font_path_cache[f]
    def _get_font(self,f,s):k=(f.lower(),s);p=self._font_cache.get(k);return p if p else self._load_font(f,s,k)
    def _load_font(self,f,s,k):fp=self._find_font_path(f);self._font_cache[k]=ImageFont.truetype(fp,s) if fp else ImageFont.load_default();return self._font_cache[k]
    def _create_text_texture(self,t,f,s,c):
        font=self._get_font(f,s);k=(t,font.path if hasattr(font,'path') else 'default',s,c);p=self._text_cache.get(k)
        if p: return p
        tid, w, h = self._render_text_texture(t,font,c,k)
        self._text_cache[k] = (tid, w, h)
        return tid, w, h
    def _render_text_texture(self,t,font,c,k):
        b=font.getbbox(t);w,h=b[2]-b[0],b[3]-b[1];i=Image.new('RGBA',(b[2],b[3]),(0,0,0,0));d=ImageDraw.Draw(i);d.text((-b[0],-b[1]),t,font=font,fill=c+(255,));tid=self._create_texture_from_image(i);return tid, i.width, i.height
    def _create_texture_from_image(self,i):
        d=i.tobytes();tid=glGenTextures(1);glBindTexture(GL_TEXTURE_2D,tid);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,i.width,i.height,0,GL_RGBA,GL_UNSIGNED_BYTE,d);self._texture_dims[tid]=(i.width,i.height);return tid
    def _load_texture_file(self,p):
        i=Image.open(p).convert("RGBA");tid=self._create_texture_from_image(i);self._textures[p]=tid;return tid

# --- GLFWキー定数 ---
KEY_A=glfw.KEY_A;KEY_B=glfw.KEY_B;KEY_C=glfw.KEY_C;KEY_D=glfw.KEY_D;KEY_E=glfw.KEY_E;KEY_F=glfw.KEY_F;KEY_G=glfw.KEY_G;KEY_H=glfw.KEY_H;KEY_I=glfw.KEY_I;KEY_J=glfw.KEY_J;KEY_K=glfw.KEY_K;KEY_L=glfw.KEY_L;KEY_M=glfw.KEY_M;KEY_N=glfw.KEY_N;KEY_O=glfw.KEY_O;KEY_P=glfw.KEY_P;KEY_Q=glfw.KEY_Q;KEY_R=glfw.KEY_R;KEY_S=glfw.KEY_S;KEY_T=glfw.KEY_T;KEY_U=glfw.KEY_U;KEY_V=glfw.KEY_V;KEY_W=glfw.KEY_W;KEY_X=glfw.KEY_X;KEY_Y=glfw.KEY_Y;KEY_Z=glfw.KEY_Z
KEY_0=glfw.KEY_0;KEY_1=glfw.KEY_1;KEY_2=glfw.KEY_2;KEY_3=glfw.KEY_3;KEY_4=glfw.KEY_4;KEY_5=glfw.KEY_5;KEY_6=glfw.KEY_6;KEY_7=glfw.KEY_7;KEY_8=glfw.KEY_8;KEY_9=glfw.KEY_9
KEY_UP=glfw.KEY_UP;KEY_DOWN=glfw.KEY_DOWN;KEY_LEFT=glfw.KEY_LEFT;KEY_RIGHT=glfw.KEY_RIGHT
KEY_SPACE=glfw.KEY_SPACE;KEY_ENTER=glfw.KEY_ENTER;KEY_ESCAPE=glfw.KEY_ESCAPE;KEY_TAB=glfw.KEY_TAB;KEY_BACKSPACE=glfw.KEY_BACKSPACE
MOUSE_BUTTON_LEFT=glfw.MOUSE_BUTTON_LEFT;MOUSE_BUTTON_RIGHT=glfw.MOUSE_BUTTON_RIGHT;MOUSE_BUTTON_MIDDLE=glfw.MOUSE_BUTTON_MIDDLE