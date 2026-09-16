import sys
import os

# 1. 确保能引用到 OpenCOOD 的主框架
opencood_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../OpenCOOD-main'))
sys.path.insert(0, opencood_path)

# 2. 将当前目录也加入环境变量，以允许导入我们单独提取出来的 attfuse_code
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 3. 动态替换 OpenCOOD 的模型注册，强制让它使用我们 cjmnet/attfuse_code 里面的模块！
import attfuse_code.point_pillar_intermediate as local_ppi
import sys
sys.modules['opencood.models.point_pillar_intermediate'] = local_ppi

# 4. 导入 OpenCOOD 标准训练流程并执行
import argparse
from opencood.tools import train

def main():
    # 强制修改系统参数，指定为我们拷贝过来的 attfuse_config.yaml
    yaml_path = os.path.join(os.path.dirname(__file__), 'attfuse_config.yaml')
    sys.argv = ['train.py', '--hypes_yaml', yaml_path]
    
    # 开始训练！
    print("=" * 60)
    print(f"🚀 CJMNET - AttFuse Training Started!")
    print(f"📝 Config File: {yaml_path}")
    print(f"🧠 Local Model Loaded: attfuse_code.point_pillar_intermediate")
    print("=" * 60)
    
    train.main()

if __name__ == '__main__':
    main()
