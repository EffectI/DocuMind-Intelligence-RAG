# unset HSA_OVERRIDE_GFX_VERSION
import torch
import sys
import os

print("========== 환경 진단 시작 ==========")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# 핵심: 우리가 설정한 환경 변수가 잘 들어갔는지 확인
print(f"\n[환경 변수 확인]")
print(f"HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', '설정안됨(Not Set)')}")
print(f"ROCM_PATH: {os.environ.get('ROCM_PATH', '설정안됨(Not Set)')}")

print(f"\n[GPU 연결 테스트]")
try:
    # GPU 사용 가능 여부 확인
    is_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {is_available}")

    if is_available:
        print(f"Make/Model: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        
        # 실제 텐서 연산 테스트 (메모리에 올리기)
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"Tensor Test: 성공! (값: {x})")
    else:
        print("❌ 실패: GPU를 인식하지 못했습니다.")
        
except Exception as e:
    print(f"❌ 에러 발생: {e}")

print("====================================")