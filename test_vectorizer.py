import VulkanMatrixMultiply
import numpy as np
import time
import os

# --- 定义一个大型计算任务 ---
N_DIM = 1024
M_DIM = 1024
P_DIM = 1024

# --- 1. 初始化阶段 (只做一次！) ---
print("="*50)
print("      🚀 最终 Tensorized Shader 性能测试 🚀")
print("="*50)
print("\n[初始化阶段] 正在创建并预分配Vulkan引擎资源...")
start_time_init = time.time()

# 找到着色器文件路径
# We are now using our new, high-performance tiled shader
shader_file_path = "comp_tensorized.spv" # <--- CHANGE TO THIS
if not os.path.exists(shader_file_path):
    print(f"❌ 错误: 着色器文件 '{shader_file_path}' 未找到！")
    print("   请先手动运行: glslc -fshader-stage=compute shaders/comp_tiled.glsl -o comp_tiled.spv")
    exit()

# 初始化引擎，传入最大维度和着色器路径
engine = VulkanMatrixMultiply.VulkanEngine(N_DIM, M_DIM, P_DIM, shader_file_path)

end_time_init = time.time()
print(f"✅ Vulkan引擎初始化完毕! 耗时: {(end_time_init - start_time_init)*1000:.2f} ms")


# --- 2. 模拟多次计算 (这才是真实世界的使用场景) ---
iterations = 10
gpu_durations = []
cpu_durations = []

print(f"\n[计算阶段] 正在模拟执行 {iterations} 次大规模矩阵乘法...")

for i in range(iterations):
    print(f" -> 计算第 {i+1}/{iterations} 次...")
    
    # 每次都创建新的随机数据
    matrix_a = np.random.rand(N_DIM, M_DIM).astype(np.float32).flatten().tolist()
    matrix_b = np.random.rand(M_DIM, P_DIM).astype(np.float32).flatten().tolist()

    # --- GPU 计算 ---
    start_time_gpu = time.time()
    result_gpu = engine.multiply(matrix_a, matrix_b, N_DIM, M_DIM, P_DIM)
    end_time_gpu = time.time()
    gpu_durations.append((end_time_gpu - start_time_gpu) * 1000)

    # --- CPU 计算 (作为对比) ---
    # We will skip the CPU calculation verification for this performance test
    # to focus on the GPU improvement. If you need to verify correctness,
    # you can uncomment this section.
    # start_time_cpu = time.time()
    # matrix_a_np = np.array(matrix_a, dtype=np.float32).reshape(N_DIM, M_DIM)
    # matrix_b_np = np.array(matrix_b, dtype=np.float32).reshape(M_DIM, P_DIM)
    # result_cpu = (matrix_a_np @ matrix_b_np).flatten().tolist()
    # end_time_cpu = time.time()
    # cpu_durations.append((end_time_cpu - start_time_cpu) * 1000)

# --- 3. 结果总结 ---
avg_gpu = sum(gpu_durations) / iterations
# avg_cpu = sum(cpu_durations) / iterations if cpu_durations else 0
# speedup = avg_cpu / avg_gpu if avg_gpu > 0 and avg_cpu > 0 else float('inf')

print("\n" + "="*50)
print("      📊 性能总结 (多次运行平均值) 📊")
print("="*50)
print(f"  - 平均GPU计算耗时 (Tiled Shader): {avg_gpu:.4f} ms")
# print(f"  - 平均CPU计算耗时: {avg_cpu:.4f} ms")
# print("\n" + "#"*50)
# print(f"🚀 Tiled Shader GPU 的速度是 Numpy CPU 的 {speedup:.2f} 倍！🚀")
# print("#"*50)
