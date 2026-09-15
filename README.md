# SANA 1.5 4.8B: English caption–image LoRA

영어 `imageCaption`과 대응 이미지를 사용해 **SANA 1.5 4.8B 교사 모델의 LoRA**를
학습하고, 저장한 어댑터를 불러 영어 프롬프트에서 이미지를 생성합니다.
기본 모델은 이전 한국어·영어 캡션 비교에 사용한
`Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers`이며,
revision은 `9468102c3cebb657f8c4b5f1e5a71e989a15f10d`로 고정합니다.

이번 작업은 **BF16 기본 transformer + LoRA 미세조정**입니다.
4비트 QLoRA나 Student 증류를 구현한 패키지는 아닙니다. 경량 Student는 교사 학습 후
별도 증류 실험에서 다룹니다. 이전 1.6B Teacher + 600M Student 스모크와의 관계는
[이전 실험 기록](docs/PREVIOUS_EXPERIMENTS.md)에 정리했습니다.

## 입력 형식

파일명까지 일치하는 이미지와 JSON을 한 폴더에 둡니다.

```text
english_pairs/
  03_01T_01S_9788959991006_57265.jpg
  03_01T_01S_9788959991006_57265.json
  ...
```

각 JSON의 키는 하나입니다. 다음 문장은 형식 설명용 예시입니다.

```json
{"imageCaption": "A small rabbit holds a yellow umbrella beside a pond."}
```

캡션 번역은 이 저장소 실행 전에 준비합니다. 이 코드는 번역 API를 호출하지 않습니다.
QA·줄거리·등장인물 태그는 SANA 조건에 추가하지 않습니다. 기본 스타일 접두어도
빈 문자열입니다. ISBN과 선택적으로 전달한 원본 메타데이터는 데이터 분리에만 씁니다.

## 설치

Python 3.11, NVIDIA CUDA GPU를 사용합니다. 아래 명령은 새 전용 환경에서 실행합니다.
PyTorch 2.5.1 / CUDA 12.1 환경에서 검증하며, 회사 서버의 CUDA 드라이버 호환성을 확인합니다.

```bash
python -m venv .venv
# Linux/WSL:
source .venv/bin/activate
# Windows PowerShell에서는: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python preflight.py
```

처음 실행할 때 Hugging Face에서 공식 모델을 내려받습니다. 이미 모델을 받은 환경에서는
`cache_features.py`와 `generate.py`에 `--local-files-only`를 지정할 수 있습니다.
이 패키지에 NVlabs/Sana 전체 저장소 설치나 기존 경량 증류 환경 설치를 덧붙일 필요는 없습니다.

## 1. 영어 이미지 쌍 준비

```bash
python prepare_pairs.py --pair-dir /data/english_pairs --output-dir data --expected-count 40001
```

Windows에서는 `/data/english_pairs` 대신 실제 폴더 경로를 인용부호로 감쌉니다.
파일명 누락·중복·연결 오류·빈 캡션·한글 잔존·깨진 이미지가 있으면 `report.json`을
기록하고 중단합니다. 누락된 영어를 한국어로 대신 채우지 않습니다.
`--expected-count` 기본값은 40,001이며 작은 명시적 시험 데이터에서는 그 수를 전달합니다.

모든 이미지–캡션 행을 유지합니다. 같은 이미지에 서로 다른 캡션이 있는 경우도 보존하되
같은 split에 배치합니다. ISBN-10/13으로 같은 책을 연결하고, 이미지 바이트·픽셀 중복을
묶어 train/validation/test를 분리합니다. 기본 비율 80/10/10은 **책 그룹의 기대 비율**이며
실제 행 비율은 `data/report.json`을 확인합니다. 비슷하지만 해시가 다른 이미지까지 탐지하지는 않습니다.

기존 LLM 평가와 도서 분리를 공유하려면 다음 옵션을 함께 전달합니다.

```bash
python prepare_pairs.py --pair-dir /data/english_pairs --output-dir data --expected-count 40001 --source-manifest /data/source_manifest.jsonl --existing-split-dir /data/qwen_splits
```

`qwen_splits`에는 `train.jsonl`, `validation.jsonl`, `test.jsonl`이 필요합니다.
원본 메타데이터와 충돌하는 분할은 test > validation > train 우선순위를 적용해 기록합니다.
원본 이미지·라벨은 수정하지 않습니다. 회사 PC로 데이터를 옮긴 뒤에는 해당 PC 경로로
`prepare_pairs.py`를 다시 실행합니다.

## 2. 실행 스모크: 짧은 학습 → 저장 → 새 프로세스 재개 → 생성

```bash
python run_smoke.py --data-dir data --output runs/teacher_smoke --resolution 512
```

학습 최대 4건·검증 최대 2건을 캐시하고, 1 optimizer step 후 종료해 checkpoint에서
새 프로세스로 재개하여 총 2 step을 실행합니다. 같은 검증 캡션과 seed로 기본 모델과
LoRA 모델의 이미지를 각각 생성합니다. 결과는
`runs/teacher_smoke/validation.json`과 `runs/teacher_smoke/compare/index.html`입니다.
명령만 확인하려면 `--dry-run`을 추가합니다.

이 시험은 연결·메모리·수치·저장·재개 검사입니다. 2 step 학습으로 그림 품질이나
캡션 정확도가 좋아졌다는 결론을 내리지 않습니다. 큰 실제 모델과 작은 CPU 테스트의
검증 범위를 [VALIDATION.md](VALIDATION.md)에 별도로 기록합니다.

## 3. 파일럿 학습

```bash
python cache_features.py --data-dir data --cache-dir cache/pilot --resolution 512 --train-limit 512 --validation-limit 32
python train_lora.py --cache cache/pilot/latest.json --output runs/pilot --max-steps 100 --learning-rate 0.0001
python sample_compare.py --cache cache/pilot/latest.json --adapter runs/pilot/final_adapter --output runs/pilot_compare --count 8 --seeds 17,23
```

텍스트 인코더와 VAE를 순서대로 사용해 특징을 캐시한 후, 학습에서는 transformer만
GPU에 올립니다. 기본 batch 1, gradient accumulation 4, LoRA rank 8,
gradient checkpointing을 사용합니다. VAE는 처음부터 FP32로 읽어 캐시와 생성의
가중치 정밀도를 일치시킵니다. `--use-8bit-adam`은 옵티마이저만 줄이며 QLoRA가 아닙니다.

이미지는 EXIF 방향을 반영하고 종횡비를 유지한 채 정사각형에 흰 여백을 넣습니다.
좌우 반전·무작위 crop으로 위치나 등장인물을 바꾸지 않습니다. 이 전처리는 여백도
학습하므로 최종 활용 이미지에서 확인해야 합니다. 실제 SANA 토크나이저로 300토큰을
넘으면 조용히 자르지 않고 중단합니다. 옵션과 데이터가 바뀌면 다른 캐시를 만듭니다.

## 4. 전체 train split 학습

회사 GPU에서 사용할 시작 예시입니다. 메모리·시간은 해당 장비의 파일럿으로 확인합니다.

```bash
python cache_features.py --data-dir data --cache-dir cache/full --resolution 1024
python train_lora.py --cache cache/full/latest.json --output runs/teacher_full --epochs 1 --learning-rate 0.00005 --warmup-steps 100
python sample_compare.py --cache cache/full/latest.json --adapter runs/teacher_full/final_adapter --output runs/full_validation --count 16 --seeds 17,23
```

제한 없는 train 캐시의 `--epochs 1`은 **train split 전체를 한 번 사용**합니다.
validation/test는 학습하지 않으므로 40,001개 전부를 optimizer에 넣는다는 의미는 아닙니다.
학습용 캐시에 `--train-limit`가 적용됐다면 `--epochs`를 거절합니다.
검증 flow MSE는 손실 진단이며 이미지의 의미 정확도 점수가 아닙니다.
하이퍼파라미터 선택은 validation으로 수행하고 최종 test는 남겨 둡니다.

중단 시 원래 옵션과 출력 폴더를 그대로 사용해 복원합니다.

```bash
python train_lora.py --cache cache/full/latest.json --output runs/teacher_full --epochs 1 --learning-rate 0.00005 --warmup-steps 100 --resume runs/teacher_full/checkpoint-50
```

메모리가 부족하면 기록을 확인하고 해상도를 낮춰 새 캐시를 생성하거나 회사의 더 큰 GPU에서
실행합니다. 모델을 자동으로 작은 버전으로 바꾸지 않습니다. 12GB에서 추론이 됐다는 사실만으로
1024 해상도 학습도 가능하다고 간주하지 않습니다.

## 5. LLM이 만든 영어 캡션으로 이미지 생성

```bash
python generate.py --adapter runs/teacher_full/final_adapter --prompt "An elephant at the zoo holds a yellow balloon with its trunk." --output runs/new_scene
```

개별 매칭 JSON도 바로 입력할 수 있습니다.

```bash
python generate.py --adapter runs/teacher_full/final_adapter --caption-json /data/english_pairs/03_01T_01S_9788959991006_57265.json --output runs/one_caption
```

LLM 출력 여러 건은 `{"id":"scene_001","imageCaption":"English caption"}` 형식의 JSONL로 저장합니다.
`examples/prompts.jsonl`은 형식 예시이며 원본 학습 데이터가 아닙니다.

```bash
python generate.py --adapter runs/teacher_full/final_adapter --prompts-jsonl examples/prompts.jsonl --output runs/llm_prompts
```

어댑터를 생략하면 기본 모델로 생성합니다. 어댑터를 넣으면 모델 revision·학습 해상도·접두어·
토큰 한도를 학습 설정에서 가져옵니다. 호환되지 않는 기본 모델을 결합하면 거절합니다.
생성에는 대응 원본 이미지를 입력하지 않습니다. 출력 PNG, 사용한 캡션, seed, 모델/어댑터 해시,
시간과 메모리를 기록합니다.

## 코드 검사와 공유

```bash
python -m unittest discover -s . -p "test_*.py" -v
python package_code.py
```

원본 데이터, 실제 시험 입력/이미지, 특징 캐시, 모델 가중치, 어댑터, API 키는 GitHub/소스 ZIP에
포함하지 않습니다. 별도 `requirements.txt`와 소스만으로 실행 경로를 제공합니다.
라이선스는 코드에 적용되며 모델과 데이터의 이용 조건은 원 배포처를 따릅니다.

참고: [Diffusers SanaPipeline](https://huggingface.co/docs/diffusers/api/pipelines/sana),
[SANA LoRA 문서](https://nvlabs.github.io/Sana/docs/sana_lora_dreambooth/).
