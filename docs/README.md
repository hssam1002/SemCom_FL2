# 문서 (Documentation)

이 폴더에는 프로젝트의 상세 문서들이 포함되어 있습니다.

## 문서 목록

### 빠른 시작
- **README_TRAINING.md**: Training 가이드 (COCO dataset, training 실행 방법)
- **PROJECT_STRUCTURE.md**: 프로젝트 구조 상세 설명

### 아키텍처 및 설계
- **TRAINING_ARCHITECTURE.md**: Training 아키텍처 (Frozen vs Trainable components)
- **COMPRESSION_POINT_COMPARISON.md**: 압축 지점 비교 분석 (Method 1 vs Method 2)
- **tx_rx_separation_analysis.md**: Tx/Rx 분리 방법 상세 분석

### Tokenization 관련
- **tokenization_difference_summary.md**: Tokenization 차이 요약 (최종 정리) ⭐
- **tokenization_difference_explanation.md**: Tokenization 차이 상세 설명
- **tokenization_analysis.md**: Tokenization 차이 분석 (초기 분석)

### Text Embedding 관련
- **TEXT_EMBEDDING_SHARING_UPDATE.md**: Text embedding 공유 구조 업데이트 내역
- **dummy_image_text_embedding_result.md**: Dummy image 사용 시 text embedding 동일성 검증 결과

### 리소스
- **assets/Florence-2-Diagram.png**: Florence-2 아키텍처 다이어그램
- **florence2_structure.txt**: Florence-2 모델 구조 상세 정보

## 문서 읽는 순서 (권장)

1. **README_TRAINING.md**: Training 시작하기
2. **TRAINING_ARCHITECTURE.md**: Training 아키텍처 이해 (Frozen components)
3. **COMPRESSION_POINT_COMPARISON.md**: 압축 지점 선택 이유 이해
4. **tokenization_difference_summary.md**: Tokenization 차이 핵심 이해
5. **TEXT_EMBEDDING_SHARING_UPDATE.md**: Text embedding 공유 구조 이해

## 추가 정보

프로젝트의 메인 README는 프로젝트 루트의 `README.md`를 참조하세요.
