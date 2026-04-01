import subprocess
import os

# --- 1. 设置您能工作的 llama.cpp 程序的路径 ---
# 我们假设多模态程序叫 llama-mtmd-cli, 如果是别的名字 (如 llava-cli), 请修改
LLAMA_CLI_PATH = "/home/weiyubin/llama.cpp/build-vulkan-new/bin/llama-mtmd-cli"

# --- 2. 设置您的模型和图片路径 (使用 2B 模型进行测试) ---
MODEL_PATH = "/home/weiyubin/models/Qwen3-VL-2B-Thinking-1M-BF16.gguf" # 请确认这是您 2B 模型的文件名
MMPROJ_PATH = "/home/weiyubin/models/Qwen3-VL-2B-Thinking-1M-mmproj-BF16.gguf" # 请确认这是配套的 mmproj 文件名
IMAGE_PATH = os.path.abspath("./images/粘贴的图像.png") # 获取图片的绝对路径

# --- 3. 您的文本问题 ---
user_prompt = "这张图片里的代码是做什么的？"

# --- 4. 构建并执行命令行指令 ---
# 这是最关键的一步: 我们在代码里拼装出一个您在终端里就能成功运行的命令
command = [
    LLAMA_CLI_PATH,
    "-m", MODEL_PATH,
    "--mmproj", MMPROJ_PATH,
    "--image", IMAGE_PATH,
    "-p", user_prompt,
    "-ngl", "99",        # 使用您的 GPU
    "--jinja"           # 使用您提供的技术文档中最重要的参数
]

print("正在通过命令行执行 llama.cpp...")
print(f"执行的命令是: {' '.join(command)}")

try:
    # 使用 subprocess 模块来运行这个命令, 并捕获它的输出
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True  # 如果命令执行失败, 会直接抛出异常
    )

    # --- 5. 打印模型的回答 ---
    # 模型的回答会直接打印在标准输出中
    print("\n模型回答:")
    print(result.stdout.strip())

except FileNotFoundError:
    print(f"错误: 找不到可执行文件 '{LLAMA_CLI_PATH}'")
    print("请确认该路径是否正确, 以及该文件是否存在。")
except subprocess.CalledProcessError as e:
    print("\n命令执行失败!")
    print(f"返回码: {e.returncode}")
    print("\n--- 错误输出 (stderr) ---")
    print(e.stderr)
