import numpy as np
import math
import random

# --- Kivy Imports ---
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics import Callback
from kivy.config import Config

# --- 强制配置 Kivy ---
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '800')
Config.set('graphics', 'multisamples', '2')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

from kivy.core.window import Window

# --- OpenGL Imports ---
from OpenGL.GL import *
from OpenGL.GLU import *

# ==========================================
#   配置与算法区域 (保持不变)
# ==========================================
NUM_LEAF_PARTICLES = 45000 
NUM_RIBBON_PARTICLES = 8000
NUM_ORNAMENTS = 2500 
NUM_FILLER_PARTICLES = 25000
NUM_BG_STARS = 40000 
NUM_TOP_STAR_PARTICLES = 5000 

MOUSE_SENSITIVITY = 0.3
TRANSITION_SPEED = 0.02 
FLOAT_SPEED = 2.0 
FLOAT_AMP = 0.06 
TREE_HEIGHT = 4.5
TREE_BASE_RADIUS = 2.0

COLOR_LEAF_BASE = (0.2, 0.0, 0.3)
COLOR_LEAF_TIP = (0.6, 0.2, 0.8)
COLOR_TOP_STAR = (0.9, 0.8, 1.0)
COLOR_RIBBON = (0.5, 0.4, 0.9)     
ORNAMENT_COLORS = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.5, 1.0), (1.0, 0.9, 0.2)]
COLOR_FILLER_BASE = (0.8, 0.7, 0.8)
COLOR_FILLER_TIP = (1.0, 0.9, 0.9)

def calculate_math_flower_pos(count, p_type):
    target_pos = np.zeros((count, 3), dtype=np.float32)
    SCALE = 2.2 

    def get_petal_pos(n, fill_mode=False):
        num_petals = 12
        petal_idx = np.random.randint(0, num_petals, n)
        t = np.random.uniform(0, 1, n)
        angle_offset = 2 * np.pi * petal_idx / num_petals

        if fill_mode:
            fill_factor = np.random.uniform(0.15, 1.0, n) ** 0.5
        else:
            fill_factor = 1.0

        r_raw = 1.5 * np.sqrt(t) * fill_factor
        theta = 1.2 * np.pi * t + angle_offset
        
        x_math = r_raw * np.cos(theta)
        y_math = r_raw * np.sin(theta) 

        r_norm = r_raw / 1.5
        base_cup_height = 1.0 * (r_raw ** 2.4)
        amp_mod = 1.0 - 0.7 * r_norm
        phase_shift = 5.0 * r_norm
        wave_height = 1.2 * np.sin(7 * theta + phase_shift) * amp_mod
        z_math_combined = (base_cup_height + wave_height) * (0.8 + 0.2 * fill_factor)
        z_math_final = z_math_combined - 2.0
        
        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 0] = x_math * SCALE
        pos[:, 1] = z_math_final * SCALE
        pos[:, 2] = y_math * SCALE
        
        if fill_mode: pos[:, 1] += 0.08 
        return pos

    if p_type == 'leaf':
        target_pos = get_petal_pos(count, fill_mode=False)
    elif p_type == 'filler':
        target_pos = get_petal_pos(count, fill_mode=True)
    elif p_type == 'ribbon':
        target_pos = get_petal_pos(count, fill_mode=False)
        target_pos *= 1.05
        target_pos[:, 1] += 0.15
    elif p_type == 'ornament':
        target_pos = get_petal_pos(count, fill_mode=False)
        target_pos[:, 1] += 0.25 
        target_pos += np.random.uniform(-0.15, 0.15, target_pos.shape)
    elif p_type == 'star':
        r_raw = np.random.rand(count) * 0.4 
        theta = np.random.rand(count) * 2 * np.pi
        x_math = r_raw * np.cos(theta)
        y_math = r_raw * np.sin(theta)
        z_math = np.random.rand(count) * 0.3 + 0.1
        target_pos[:, 0] = x_math * SCALE
        target_pos[:, 1] = z_math * SCALE
        target_pos[:, 2] = y_math * SCALE

    return target_pos.astype(np.float32)

def generate_geometry():
    print("Generating geometry...")
    # 1. 树叶
    h_ratios = np.random.power(1.5, NUM_LEAF_PARTICLES)
    heights = -2.0 + h_ratios * TREE_HEIGHT
    radii = TREE_BASE_RADIUS * (1 - h_ratios)
    radii += np.random.uniform(-0.1, 0.1, NUM_LEAF_PARTICLES)
    radii = np.maximum(0, radii)
    thetas = np.random.uniform(0, 2 * np.pi, NUM_LEAF_PARTICLES)
    x, z = radii * np.cos(thetas), radii * np.sin(thetas)
    leaves_pos_tree = np.stack((x, heights, z), axis=-1).astype(np.float32)
    leaves_pos_flower = calculate_math_flower_pos(NUM_LEAF_PARTICLES, 'leaf')
    
    colors = np.zeros((NUM_LEAF_PARTICLES, 3), dtype=np.float32)
    for i in range(3):
        colors[:, i] = COLOR_LEAF_BASE[i] * (1 - h_ratios) + COLOR_LEAF_TIP[i] * h_ratios
    colors += np.random.uniform(-0.05, 0.05, colors.shape)
    leaves_col = np.clip(colors, 0, 1).astype(np.float32)
    leaves_phase = np.random.uniform(0, 2*np.pi, (NUM_LEAF_PARTICLES, 3)).astype(np.float32)

    # 2. 填充粒子
    filler_heights = np.random.uniform(-2.0, TREE_HEIGHT-2.0, NUM_FILLER_PARTICLES)
    filler_radii = np.random.uniform(0, 0.3, NUM_FILLER_PARTICLES)
    filler_thetas = np.random.uniform(0, 2 * np.pi, NUM_FILLER_PARTICLES)
    fx, fz = filler_radii * np.cos(filler_thetas), filler_radii * np.sin(filler_thetas)
    filler_pos_tree = np.stack((fx, filler_heights, fz), axis=-1).astype(np.float32)
    filler_pos_flower = calculate_math_flower_pos(NUM_FILLER_PARTICLES, 'filler')

    f_ratios = (filler_heights + 2.0) / TREE_HEIGHT
    filler_colors = np.zeros((NUM_FILLER_PARTICLES, 3), dtype=np.float32)
    for i in range(3):
        filler_colors[:, i] = COLOR_FILLER_BASE[i] * (1 - f_ratios) + COLOR_FILLER_TIP[i] * f_ratios
    filler_col = np.clip(filler_colors, 0, 1).astype(np.float32)
    filler_phase = np.random.uniform(0, 2*np.pi, (NUM_FILLER_PARTICLES, 3)).astype(np.float32)

    # 3. 彩带
    rb_cnt = NUM_RIBBON_PARTICLES
    rb_prog = np.linspace(0, 1, rb_cnt)
    rb_angle = rb_prog * 8.0 * 2 * np.pi
    rb_y = -1.9 + rb_prog * (TREE_HEIGHT - 0.2)
    rb_r = (TREE_BASE_RADIUS + 0.1) * (1 - rb_prog) * 1.05
    rb_x = rb_r * np.cos(rb_angle)
    rb_z = rb_r * np.sin(rb_angle)
    jitter = np.random.uniform(-0.08, 0.08, (rb_cnt, 3)) 
    ribbon_pos_tree = np.stack((rb_x, rb_y, rb_z), axis=-1).astype(np.float32) + jitter
    ribbon_pos_flower = calculate_math_flower_pos(rb_cnt, 'ribbon')
    ribbon_col = np.tile(COLOR_RIBBON, (rb_cnt, 1)).astype(np.float32)
    ribbon_phase = np.random.uniform(0, 2*np.pi, (rb_cnt, 3)).astype(np.float32)

    # 4. 装饰灯
    orn_cnt = NUM_ORNAMENTS
    indices = np.random.choice(NUM_LEAF_PARTICLES, orn_cnt, replace=False)
    orn_pos_tree = leaves_pos_tree[indices] * 1.08
    orn_pos_flower = calculate_math_flower_pos(orn_cnt, 'ornament')
    orn_col = np.array([random.choice(ORNAMENT_COLORS) for _ in range(orn_cnt)], dtype=np.float32)
    orn_phase = np.random.uniform(0, 2*np.pi, (orn_cnt, 3)).astype(np.float32)

    # 5. 树顶星
    ts_cnt = NUM_TOP_STAR_PARTICLES
    ts_theta = np.random.uniform(0, 2*np.pi, ts_cnt)
    ts_phi = np.random.uniform(0, np.pi, ts_cnt)
    ts_r = 0.3 * (np.random.random(ts_cnt) ** 8) 
    ts_x = ts_r * np.sin(ts_phi) * np.cos(ts_theta)
    ts_y = (-2.0 + TREE_HEIGHT) + ts_r * np.sin(ts_phi) * np.sin(ts_theta)
    ts_z = ts_r * np.cos(ts_phi)
    star_pos_tree = np.stack((ts_x, ts_y, ts_z), axis=-1).astype(np.float32)
    star_pos_flower = calculate_math_flower_pos(ts_cnt, 'star')
    star_col = np.tile(COLOR_TOP_STAR, (ts_cnt, 1)).astype(np.float32)

    # 6. 背景
    bg_cnt = NUM_BG_STARS
    bg_pos = np.random.uniform(-1, 1, (bg_cnt, 3)).astype(np.float32)
    bg_pos = bg_pos / np.linalg.norm(bg_pos, axis=1)[:, np.newaxis] * np.random.uniform(60, 120, (bg_cnt, 1))
    bg_col_base = np.ones((bg_cnt, 3), dtype=np.float32) * 0.25 

    return {
        'leaves': {'tree': leaves_pos_tree, 'flower': leaves_pos_flower, 'col': leaves_col, 'phase': leaves_phase},
        'filler': {'tree': filler_pos_tree, 'flower': filler_pos_flower, 'col': filler_col, 'phase': filler_phase},
        'star': {'tree': star_pos_tree, 'flower': star_pos_flower, 'col': star_col},
        'ribbon': {'tree': ribbon_pos_tree, 'flower': ribbon_pos_flower, 'col': ribbon_col, 'phase': ribbon_phase},
        'ornament': {'tree': orn_pos_tree, 'flower': orn_pos_flower, 'col': orn_col, 'phase': orn_phase},
        'bg': {'pos': bg_pos, 'col_base': bg_col_base}
    }

# ==========================================
#   Kivy 3D 渲染组件
# ==========================================

class ParticleTreeWidget(Widget):
    def __init__(self, **kwargs):
        super(ParticleTreeWidget, self).__init__(**kwargs)
        
        self.data = generate_geometry()
        print('''
        圣诞快乐哟静静
        后面那玩意儿有点丑丑的，这次没招了，下次一定
        嘻嘻
        (哼，让你不开电脑，给你搞个安卓的)
        ''')

        self.mix_factor = 0.0
        self.target_factor = 0.0
        self.rot_x = 20.0
        self.rot_y = 0.0
        self.time_counter = 0.0
        
        # 预分配计算内存
        # 【修复】统一变量名 current_ornament_pos 以匹配 update 循环
        self.current_leaves_pos = np.zeros_like(self.data['leaves']['tree'])
        self.current_filler_pos = np.zeros_like(self.data['filler']['tree'])
        self.current_ribbon_pos = np.zeros_like(self.data['ribbon']['tree'])
        self.current_ornament_pos = np.zeros_like(self.data['ornament']['tree']) 
        self.current_star_pos = np.zeros_like(self.data['star']['tree'])

        with self.canvas:
            self.cb = Callback(self.draw_gl_content)
        
        Clock.schedule_interval(self.update_gl, 1.0 / 60.0)

    def update_gl(self, dt):
        self.time_counter += dt
        diff = self.target_factor - self.mix_factor
        if abs(diff) > 0.0005:
            self.mix_factor += diff * TRANSITION_SPEED
        else:
            self.mix_factor = self.target_factor
        self.cb.ask_update()

    def _update_particle_positions(self):
        mix = self.mix_factor
        time = self.time_counter

        # 更新所有粒子位置 (Lerp + Sin Wave)
        # 此处使用 getattr 自动获取 self.current_{key}_pos
        for key in ['leaves', 'filler', 'ribbon', 'ornament']:
            np.subtract(self.data[key]['flower'], self.data[key]['tree'], out=getattr(self, f'current_{key}_pos'))
            np.multiply(getattr(self, f'current_{key}_pos'), mix, out=getattr(self, f'current_{key}_pos'))
            np.add(getattr(self, f'current_{key}_pos'), self.data[key]['tree'], out=getattr(self, f'current_{key}_pos'))
            getattr(self, f'current_{key}_pos')[:] += np.sin(time * FLOAT_SPEED + self.data[key]['phase']) * FLOAT_AMP

        # Star (无浮动)
        np.subtract(self.data['star']['flower'], self.data['star']['tree'], out=self.current_star_pos)
        np.multiply(self.current_star_pos, mix, out=self.current_star_pos)
        np.add(self.current_star_pos, self.data['star']['tree'], out=self.current_star_pos)

    def draw_gl_content(self, instr):
        # 1. 保存 Kivy 状态并停用 Shader
        #glPushAttrib(GL_ALL_ATTRIB_BITS)
        #glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS)
        glUseProgram(0) 
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

        self._update_particle_positions()
        
        # 2. 清屏
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_POINT_SMOOTH)
        
        # 3. 投影矩阵
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        w, h = self.width, self.height
        if h == 0: h = 1
        gluPerspective(45, w / h, 0.1, 200.0)
        
        # 4. 模型矩阵
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(0.0, -1.0, -13.0) 
        
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        
        if self.mix_factor > 0.01:
            glRotatef(self.time_counter * 5.0 * self.mix_factor, 0, 1, 0)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # --- 绘制开始 ---

        # A. 背景星光
        twinkle = 0.5 + 0.5 * np.sin(self.time_counter * 3.0 + self.data['bg']['pos'][:, 0])
        bg_colors = np.clip(self.data['bg']['col_base'] * (0.5 + twinkle[:, np.newaxis]), 0.0, 1.0)
        
        glPushMatrix() 
        glLoadIdentity()
        glTranslatef(0.0, -1.0, -13.0) 
        glVertexPointer(3, GL_FLOAT, 0, self.data['bg']['pos'])
        glColorPointer(3, GL_FLOAT, 0, bg_colors)
        glPointSize(1.2)
        glDrawArrays(GL_POINTS, 0, NUM_BG_STARS)
        glPopMatrix()

        # B. 树叶
        glVertexPointer(3, GL_FLOAT, 0, self.current_leaves_pos)
        glColorPointer(3, GL_FLOAT, 0, self.data['leaves']['col'])
        glPointSize(2.8)
        glDrawArrays(GL_POINTS, 0, NUM_LEAF_PARTICLES)

        # C. 填充粒子
        glVertexPointer(3, GL_FLOAT, 0, self.current_filler_pos)
        glColorPointer(3, GL_FLOAT, 0, self.data['filler']['col'])
        glPointSize(2.5)
        glDrawArrays(GL_POINTS, 0, NUM_FILLER_PARTICLES)

        # D. 彩带
        glVertexPointer(3, GL_FLOAT, 0, self.current_ribbon_pos)
        glColorPointer(3, GL_FLOAT, 0, self.data['ribbon']['col'])
        glPointSize(5.0)
        glDrawArrays(GL_POINTS, 0, NUM_RIBBON_PARTICLES)

        # E. 装饰灯 (使用修正后的变量名)
        glVertexPointer(3, GL_FLOAT, 0, self.current_ornament_pos)
        glColorPointer(3, GL_FLOAT, 0, self.data['ornament']['col'])
        glPointSize(5.0)
        glDrawArrays(GL_POINTS, 0, NUM_ORNAMENTS)

        # F. 树顶星
        glVertexPointer(3, GL_FLOAT, 0, self.current_star_pos)
        glColorPointer(3, GL_FLOAT, 0, self.data['star']['col'])
        glPointSize(4.0)
        glDrawArrays(GL_POINTS, 0, NUM_TOP_STAR_PARTICLES)

        # --- 恢复状态 ---
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        
        #glPopClientAttrib()
        #glPopAttrib()

    def on_touch_down(self, touch):
        if touch.button == 'left':
            if self.target_factor < 0.5:
                self.target_factor = 1.0
                self.rot_x = 60.0 
            else:
                self.target_factor = 0.0
                self.rot_x = 20.0
        if touch.button == 'right':
            touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.rot_y += touch.dx * MOUSE_SENSITIVITY
            self.rot_x += touch.dy * MOUSE_SENSITIVITY
            self.rot_x = max(-80, min(80, self.rot_x))
        return True
    
    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
        return True

class ChristmasApp(App):
    def build(self):
        return ParticleTreeWidget()

if __name__ == '__main__':
    ChristmasApp().run()