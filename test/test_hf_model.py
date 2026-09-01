# test_load.py
import time
from sentence_transformers import SentenceTransformer

start = time.time()
print("开始加载 embedding 模型...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"加载完成，耗时 {time.time() - start:.2f} 秒")
