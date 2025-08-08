#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA诊断脚本
检查GPU和PyTorch CUDA配置
"""

import torch
import sys
import subprocess
import platform

def check_python_version():
    """检查Python版本"""
    print("=== Python版本信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"平台: {platform.platform()}")
    print()

def check_pytorch_installation():
    """检查PyTorch安装信息"""
    print("=== PyTorch安装信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"PyTorch编译的CUDA版本: {torch.version.cuda if torch.version.cuda else 'None'}")
    print(f"PyTorch编译的cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'None'}")
    print()

def check_cuda_availability():
    """检查CUDA可用性"""
    print("=== CUDA可用性检查 ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n设备 {i}: {props.name}")
            print(f"  计算能力: {props.major}.{props.minor}")
            print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"  多处理器数量: {props.multi_processor_count}")
            
            # 检查内存使用情况
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  已分配内存: {memory_allocated:.2f} GB")
            print(f"  已保留内存: {memory_reserved:.2f} GB")
    else:
        print("CUDA不可用")
    print()

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("=== NVIDIA驱动检查 ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # 只显示前10行
                if 'Driver Version' in line or 'CUDA Version' in line:
                    print(line.strip())
            print("nvidia-smi命令执行成功")
        else:
            print("nvidia-smi命令执行失败")
            print(f"错误信息: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("nvidia-smi命令超时")
    except FileNotFoundError:
        print("未找到nvidia-smi命令，可能NVIDIA驱动未正确安装")
    except Exception as e:
        print(f"检查NVIDIA驱动时出错: {e}")
    print()

def check_cuda_runtime():
    """检查CUDA运行时"""
    print("=== CUDA运行时检查 ===")
    try:
        # 尝试创建一个简单的CUDA张量
        if torch.cuda.is_available():
            device = torch.device('cuda')
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.matmul(x, y)
            print("CUDA张量运算测试: 成功")
            print(f"测试张量设备: {z.device}")
            print(f"测试张量形状: {z.shape}")
        else:
            print("CUDA不可用，无法进行运行时测试")
    except Exception as e:
        print(f"CUDA运行时测试失败: {e}")
    print()

def check_environment_variables():
    """检查相关环境变量"""
    print("=== 环境变量检查 ===")
    import os
    
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES']
    for var in cuda_vars:
        value = os.environ.get(var)
        print(f"{var}: {value if value else '未设置'}")
    print()

def provide_recommendations():
    """提供建议"""
    print("=== 建议和解决方案 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，可能的原因和解决方案:")
        print("1. PyTorch安装的是CPU版本")
        print("   解决方案: 重新安装支持CUDA的PyTorch版本")
        print("   访问 https://pytorch.org/ 获取正确的安装命令")
        print()
        print("2. NVIDIA驱动未正确安装")
        print("   解决方案: 安装或更新NVIDIA显卡驱动")
        print()
        print("3. CUDA工具包版本不兼容")
        print("   解决方案: 确保CUDA版本与PyTorch版本兼容")
        print()
        print("4. 环境变量配置问题")
        print("   解决方案: 检查CUDA_HOME等环境变量设置")
    else:
        print("✅ CUDA可用，训练应该能够使用GPU")
        print("如果训练仍然使用CPU，请检查:")
        print("1. 配置文件中的device设置")
        print("2. 代码中的设备选择逻辑")
        print("3. 是否有其他进程占用GPU内存")
    print()

def main():
    """主函数"""
    print("CUDA诊断工具")
    print("=" * 50)
    print()
    
    check_python_version()
    check_pytorch_installation()
    check_cuda_availability()
    check_nvidia_driver()
    check_cuda_runtime()
    check_environment_variables()
    provide_recommendations()
    
    print("诊断完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()