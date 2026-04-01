import VulkanMatrixMultiply
import numpy as np
import time

# --- Step 1: Configuration (配置) ---
# 定义我们个人向量的维度
INPUT_DIM = 12      # C++ BoW向量的维度
OUTPUT_DIM = 1024   # 最终输出的高维向量维度

# --- Step 2: One-Time Setup (一次性设置) ---
# 在真实应用中，这个投影矩阵应该被加载，而不是每次都重新生成
print(f">> 正在加载/创建投影矩阵 ({OUTPUT_DIM} x {INPUT_DIM})...")
projection_matrix_np = np.random.rand(OUTPUT_DIM, INPUT_DIM).astype(np.float32)
print(">> 投影矩阵准备就绪。")


def generate_personal_vector(text: str) -> list[float]:
    """
    这个函数是你所有工作的核心成果。
    它接收一段文本，返回一个高维的、可用于搜索的个人向量。

    This function is the core result of all your work.
    It takes a piece of text and returns a high-dimensional, searchable personal vector.
    """
    print("\n" + "-"*50)
    print(f"输入文本: '{text}'")

    # --- Stage 1: Foundational Vectorization (基础向量化) ---
    # 调用你亲手编写的、经过验证的C++模块
    # Calling the C++ module you wrote and verified yourself.
    print("   -> [Stage 1] 正在调用 C++ 模块生成基础BoW向量...")
    low_dim_vector = VulkanMatrixMultiply.text_to_vector(text)
    print(f"   -> 基础向量 (12维): {low_dim_vector}")

    # --- Stage 2: High-Dimensional Projection (高维投影) ---
    # 使用我们通过测试证明在此任务上最高效的Numpy引擎
    # Using the Numpy engine, which we proved is most efficient for this task.
    print(f"   -> [Stage 2] 正在使用 Numpy (CPU) 将向量投影到 {OUTPUT_DIM} 维...")
    
    # 将输入转换为Numpy格式
    low_dim_vector_np = np.array(low_dim_vector, dtype=np.float32).reshape(INPUT_DIM, 1)
    
    # 执行矩阵乘法
    high_dim_vector = (projection_matrix_np @ low_dim_vector_np).flatten().tolist()
    
    print("   -> 高维投影完成！")
    
    return high_dim_vector


# --- Step 3: Main Execution (主程序入口) ---
if __name__ == "__main__":
    start_time = time.time()
    
    # 调用我们的核心功能
    my_text = "Vulkan and C++ are essential for high performance GPU code."
    final_vector = generate_personal_vector(my_text)
    
    end_time = time.time()
    
    print("\n" + "="*50)
    print("      ✅ 个人向量生成成功！ ✅")
    print("="*50)
    print(f"最终向量维度: {len(final_vector)}")
    print(f"最终向量 (前5个元素): {final_vector[:5]}")
    print(f"\n总耗时: {(end_time - start_time) * 1000:.4f} ms")
    print("\n这个向量现在可以被存入你个人的向量数据库，作为未来一切的基础。")
    print("你已经完全掌控了从文本到高维向量的整个流程。")
