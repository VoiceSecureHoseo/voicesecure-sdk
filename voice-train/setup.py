"""
처음 이 저장소를 받았을 때 한 번 실행하는 설치 스크립트

하는 일:
  1. requirements.txt 의존성 설치
  2. .env.example -> .env 복사 (이미 있으면 건드리지 않음)
  3. campplus.onnx, kss(또는 kss_mini) 같은 필수 파일/폴더가 있는지 확인하고
     없는 게 있으면 무엇을 어디서 받아야 하는지 안내

실행:
  python setup.py
"""

import os
import subprocess
import sys
import shutil

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def install_requirements():
    print("[1/3] 의존성 설치 중...")
    req_path = os.path.join(_THIS_DIR, "requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_path]
    )
    if result.returncode != 0:
        print("  의존성 설치 실패. pip 출력을 확인하세요.")
        sys.exit(1)
    print("  완료\n")


def setup_env_file():
    print("[2/3] .env 파일 확인 중...")
    env_path = os.path.join(_THIS_DIR, ".env")
    example_path = os.path.join(_THIS_DIR, ".env.example")

    if os.path.exists(env_path):
        print("  이미 .env가 있어서 그대로 둡니다.\n")
        return

    shutil.copy(example_path, env_path)
    print(f"  .env.example -> .env 로 복사했습니다.")
    print(f"  {env_path} 파일을 열어 KSS_DIR, CAMPPLUS_ONNX 등 값을 채워주세요.\n")


def check_required_files():
    print("[3/3] 필수 파일 확인 중...")

    checks = [
        ("campplus.onnx", os.path.join(_THIS_DIR, "campplus.onnx"),
         "CAM++ 화자 임베딩 모델. FunASR / CosyVoice 등에서 받아 이 폴더에 두거나 "
         ".env의 CAMPPLUS_ONNX에 경로를 지정하세요."),
        ("kss 또는 kss_mini", None,
         "학습용 음성 데이터셋 폴더. KSS(한국어 단일 화자 음성)를 받아 kss/ 폴더로 "
         "두거나 .env의 KSS_DIR에 경로를 지정하세요."),
    ]

    missing = []

    if not os.path.exists(checks[0][1]):
        missing.append(checks[0])

    has_kss = os.path.isdir(os.path.join(_THIS_DIR, "kss")) or \
              os.path.isdir(os.path.join(_THIS_DIR, "kss_mini"))
    if not has_kss:
        missing.append(checks[1])

    if not missing:
        print("  필요한 파일이 모두 있습니다.\n")
    else:
        print("  아래 파일/폴더가 없습니다. 학습을 돌리기 전에 준비해주세요.\n")
        for name, _, desc in missing:
            print(f"  - {name}")
            print(f"    {desc}")
        print()


def main():
    print("=" * 55)
    print("Voice Train 설치 스크립트")
    print("=" * 55 + "\n")

    install_requirements()
    setup_env_file()
    check_required_files()

    print("=" * 55)
    print("설치 완료. 다음 명령으로 학습을 시작할 수 있습니다.")
    print("  python train.py --epochs 100 --batch 8")
    print("=" * 55)


if __name__ == "__main__":
    main()
