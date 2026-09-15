# 이전 스모크와 이번 교사 LoRA의 관계

이번 패키지의 학습 대상은 SANA 1.5 **4.8B**입니다. 교사를 이미지–영어 캡션으로
미세조정한 뒤 경량 Student에 증류하는 후속 실험을 계획합니다.

| 실험 | 실제 모델과 범위 |
| --- | --- |
| 로컬 `sana_teacher_smoke` | SANA 1.5 4.8B, 1024px, 2 sampling step, 기본 모델 이미지 생성 |
| 로컬 한국어·영어 비교 | 같은 4.8B, 1024px, 18 sampling step, 캡션 10개 × 2언어 |
| 기존 GitHub 경량 학습 스모크 | 1.6B frozen Teacher + Sana-Sprint 600M Student + Discriminator head, 교대 갱신·저장·재개 |
| 이 저장소 `run_smoke.py` | 4.8B 교사의 caption–image LoRA 학습·재개·생성 연결 검사 |

기존 경량 **학습** 스모크와 4.8B **이미지 생성** 스모크는 별개입니다.
기존 경량 코드의 Teacher는 `Sana_1600M_1024px_BF16`, Student 초기 가중치는
`Sana_Sprint_0.6B_1024px`입니다. 해당 저장소의 기록에는 G/D 교대 10회와 재개 후
G 1회, 총 124개 검사 통과가 기록돼 있습니다. 이는 이번 4.8B LoRA의 실행 증거로
사용하지 않습니다.

이전 코드는 해당 저장소에서 그대로 재현합니다.

- [기존 스모크 실행 안내](https://github.com/wonyoungkim1996-lab/SANA_Smoke_test/blob/main/docs/teacher_student_pipeline_smoke.md)
- [이전 설정 YAML](https://github.com/wonyoungkim1996-lab/SANA_Smoke_test/blob/main/configs/sana_sprint_config/1024ms/SanaSprint_1600M_teacher_600M_student_pipeline_smoke.yaml)
- [이전 검사 결과](https://github.com/wonyoungkim1996-lab/SANA_Smoke_test/blob/main/docs/teacher_student_smoke_validation.json)

후속 증류에서 4.8B 교사를 사용하려면 경량 코드의 1.6B 교사 설정·구조와 checkpoint
로딩을 별도로 확장하고 검증해야 합니다. 모델 경로만 바꾸면 그대로 호환된다고 주장하지 않습니다.
