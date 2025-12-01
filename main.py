import taichi as ti
import numpy as np
import time 

# --- 1. 初始化 Taichi ---
ti.init(arch=ti.gpu)

# --- 2. 配置参数 (速度微调) ---
NUM_PARTICLES = 25000  
RES = (1440, 1440)  
# RES = (1280, 720) 
# 物理参数 (归一化到 0.0-1.0)
R_SMOOTH = 0.06    
R_REPEL = 0.012    
REPULSION_STR = 1.0  
FRICTION = 0.99    
DT = 0.005         
MAX_FORCE_MAG = 0.1  

# --- 3. 定义数据场 (Fields) ---
pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)  
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PARTICLES)  
types = ti.field(dtype=ti.i32, shape=NUM_PARTICLES)          
particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_PARTICLES)
interaction_matrix = ti.field(dtype=ti.f32, shape=(4, 4))
palette = ti.Vector.field(3, dtype=ti.f32, shape=4)

# --- 4. 核心计算核 (Kernel) ---

@ti.kernel
def init_particles():
    palette[0] = ti.Vector([1.0, 0.2, 0.2]) 
    palette[1] = ti.Vector([0.2, 1.0, 0.2]) 
    palette[2] = ti.Vector([0.2, 0.4, 1.0]) 
    palette[3] = ti.Vector([1.0, 1.0, 0.2]) 
    
    for i in range(NUM_PARTICLES):
        pos[i] = [ti.random(), ti.random()]
        vel[i] = [0.0, 0.0]
        t = int(ti.random() * 4)
        types[i] = t
        particle_colors[i] = palette[t]

@ti.kernel
def update_particles():
    for i in range(NUM_PARTICLES):
        total_force = ti.Vector([0.0, 0.0])
        type_i = types[i]
        
        for j in range(NUM_PARTICLES):
            if i == j: continue
            
            type_j = types[j]
            
            # --- 环绕边界计算 ---
            diff = pos[j] - pos[i]
            if diff.x > 0.5:  diff.x -= 1.0
            elif diff.x < -0.5: diff.x += 1.0
            if diff.y > 0.5:  diff.y -= 1.0
            elif diff.y < -0.5: diff.y += 1.0
            # ------------------

            dist_sq = diff.norm_sqr()
            dist = ti.sqrt(dist_sq)
            
            if dist > 0 and dist < R_SMOOTH:
                dir_vec = diff / dist 
                
                # 1. 基础引力/斥力
                g = interaction_matrix[type_i, type_j]
                force = g * (1.0 - dist / R_SMOOTH)
                total_force += force * dir_vec
                
                # 2. 短程排斥
                if dist < R_REPEL:
                    repel_force = REPULSION_STR * (1.0 - dist / R_REPEL)
                    total_force -= repel_force * dir_vec

        total_force *= 8.0  
        
        # --- 力钳制 (Force Clamping) ---
        force_mag = total_force.norm()
        if force_mag > MAX_FORCE_MAG:
            total_force *= MAX_FORCE_MAG / force_mag
        # ----------------------------------------
            
        vel[i] += total_force * DT
        vel[i] *= FRICTION
        pos[i] += vel[i] * DT
        
        # 边界修正
        pos[i].x = pos[i].x - ti.floor(pos[i].x)
        pos[i].y = pos[i].y - ti.floor(pos[i].y)

def randomize_matrix():
    new_matrix = np.random.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
    interaction_matrix.from_numpy(new_matrix)
    return new_matrix

# --- 5. 主程序 (绘图和 GUI 部分不变) ---
def main():
    init_particles()
    matrix_np = randomize_matrix()
    
    window = ti.ui.Window("GPU 粒子生命 (Taichi)", res=RES, vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()
    
    labels = ["Red", "Green", "Blue", "Yellow"]
    
    t_start = time.perf_counter()
    frame_count = 0
    fps = 0.0
    
    while window.running:
        
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        frame_count += 1
        
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            t_start = t_end
            frame_count = 0
        
        update_particles()
        
        canvas.set_background_color((0.05, 0.05, 0.05))
        canvas.circles(pos, radius=0.0015, per_vertex_color=particle_colors)
        
        # --- GUI ---
        with gui.sub_window("Controls", x=0.01, y=0.01, width=0.28, height=0.9):
            gui.text(f"Particles: {NUM_PARTICLES}")
            gui.text(f"FPS: {fps:.1f}") 
            gui.text(f"MAX_FORCE: {MAX_FORCE_MAG}") 
            
            if gui.button("Randomize Rules (R)"):
                matrix_np = randomize_matrix()
                
            if gui.button("Reset Positions (Space)"):
                init_particles()
                
            gui.text("Interaction Rules:")
            changed = False
            for i in range(4): 
                gui.text(f"--- Target: {labels[i]} ---")
                for j in range(4): 
                    old_val = float(matrix_np[i, j])
                    new_val = gui.slider_float(f"{labels[j]} -> {labels[i]}", old_val, -1.0, 1.0)
                    if new_val != old_val:
                        matrix_np[i, j] = new_val
                        changed = True
            if changed:
                interaction_matrix.from_numpy(matrix_np)
        
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                matrix_np = randomize_matrix()
            if window.event.key == ti.ui.SPACE:
                init_particles()

        window.show()

if __name__ == "__main__":
    main()
