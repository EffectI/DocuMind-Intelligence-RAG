# 1. AMD ROCm + PyTorch 공식 이미지를 베이스로 사용

FROM rocm/pytorch:rocm6.2.1_ubuntu22.04_py3.10_pytorch_release_2.3.0

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (git, 한글 폰트 등)
# (DEBIAN_FRONTEND=noninteractive는 설치 중 질문 안 나오게 하는 옵션)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사 (나중에 볼륨으로 연결하겠지만, 빌드 시점에도 넣어둠)
COPY . .

# 6. 컨테이너가 꺼지지 않게 대기 (개발용)
CMD ["tail", "-f", "/dev/null"]