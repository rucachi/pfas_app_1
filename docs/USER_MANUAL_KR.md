# ONFRA PFAS 사용자 매뉴얼

## 목차
1. [소개](#1-소개)
2. [시작하기](#2-시작하기)
3. [Feature Finding 탭](#3-feature-finding-탭)
4. [Prioritization 탭](#4-prioritization-탭)
5. [Visualization 탭](#5-visualization-탭)
6. [결과 해석 방법](#6-결과-해석-방법)
7. [문제 해결](#7-문제-해결)

---

## 1. 소개

ONFRA PFAS는 LC-MS/MS 데이터에서 PFAS(과불화화합물)를 자동으로 검출하고 우선순위를 지정하는 분석 도구입니다.

### 분석 워크플로우
```
mzML 파일 → Feature Finding → Prioritization → Visualization/Export
```

---

## 2. 시작하기

### 2.1 프로그램 실행
```
python -m onfra_pfas.app.main
```

### 2.2 필요한 입력 파일
| 파일 형식 | 설명 | 필수 여부 |
|----------|------|----------|
| `.mzML` | LC-MS/MS 원시 데이터 (centroided) | **필수** |
| `.csv/.tsv` | Suspect 리스트 (의심 물질 목록) | 선택 |

### 2.3 데이터 요구사항
- **Centroid 모드**: Profile 모드 데이터는 사전에 centroid 변환 필요
- **MS1 + MS2**: DDA 또는 DIA 모드 데이터 지원

---

## 3. Feature Finding 탭

LC-MS 데이터에서 크로마토그래픽 피크(feature)를 자동 검출합니다.

### 3.1 입력 파라미터

#### Sample mzML (필수)
| 항목 | 설명 |
|------|------|
| **파일 경로** | 분석할 mzML 파일 경로 |
| **Browse 버튼** | 파일 탐색기로 mzML 파일 선택 |

#### Parameters (Feature 검출 설정)

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| **m/z Tolerance (Da)** | 0.005 | 0.0001~0.1 | Mass trace 검출 시 m/z 허용 오차. 고해상도 기기는 0.005, 저해상도는 0.02~0.05 권장 |
| **Noise Threshold** | 1000 | 0 이상 | 노이즈 필터링 강도 임계값. 값이 클수록 감도↓, 노이즈↓ |
| **Peak SNR** | 3.0 | 1.0~20.0 | 신호 대 잡음비 (Signal-to-Noise Ratio). 3.0 권장, 감도가 필요하면 2.0 |
| **Peak FWHM (s)** | 10.0 | 1.0~300.0 | 예상 피크 폭 (반치전폭, 초 단위). UPLC는 10~20, HPLC는 30~60 |

#### GPU Acceleration

| 옵션 | 설명 |
|------|------|
| **Auto** | 데이터 크기에 따라 자동 선택 (기본값, 권장) |
| **Force GPU** | 항상 GPU 사용 (GPU 없으면 오류 발생) |
| **Force CPU** | 항상 CPU 사용 (GPU 문제 시 선택) |

#### Blank Correction (선택)

| 옵션 | 설명 |
|------|------|
| **Enable** | Blank 보정 활성화 체크박스 |
| **Policy - None** | 보정 없음 |
| **Policy - Subtract** | Blank intensity를 샘플에서 뺌 |
| **Policy - Fold Change** | Sample/Blank 비율이 3배 이상인 feature만 유지 (권장) |
| **Policy - Presence** | Blank에 존재하는 feature 모두 제거 |

### 3.2 실행 방법
1. **Browse** 버튼으로 mzML 파일 선택
2. 파라미터 조정 (기본값으로 시작 권장)
3. **Run Feature Finding** 버튼 클릭
4. Progress bar와 Log에서 진행 상황 확인

### 3.3 출력 정보
- **파일 정보**: 파일 크기, 스펙트럼 수, RT 범위
- **Feature 수**: 검출된 크로마토그래픽 피크 개수
- 자동으로 Prioritization 탭으로 전달됨

---

## 4. Prioritization 탭

검출된 Feature들에 PFAS 스코어를 부여하여 우선순위를 지정합니다.

### 4.1 입력 파라미터

#### Configuration (분석 옵션)

| 체크박스 | 기본값 | 설명 |
|---------|-------|------|
| **MD/C Filtering** | ✓ | Mass Defect/Carbon 필터링. PFAS는 특징적인 MD/C 값(-0.10~0.05) 보유 |
| **KMD Series Detection** | ✓ | Kendrick Mass Defect로 CF2 반복 단위 동족체 탐지 |
| **Diagnostic Fragment Matching** | ✓ | MS2에서 CF3⁻, C2F5⁻ 등 진단 조각 이온 매칭 |
| **Suspect Screening** | ✓ | 외부 Suspect 리스트와 m/z 매칭 |

#### 수치 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| **Suspect ppm** | 5.0 | Suspect 리스트 매칭 시 ppm 허용 오차 |
| **Min Score** | 2.0 | 최소 PFAS 스코어 임계값. 이 값 이상만 결과에 포함 |

#### Suspect List (선택)
- CSV 또는 TSV 형식의 의심 물질 목록
- 필수 컬럼: `name`, `formula` 또는 `exact_mass`
- 선택 컬럼: `cas_number`, `smiles`

### 4.2 실행 방법
1. Feature Finding 완료 후 자동으로 feature 데이터 로드됨
2. Configuration 체크박스로 분석 옵션 선택
3. **Run Prioritization** 버튼 클릭

### 4.3 결과 테이블 컬럼

| 컬럼 | 설명 |
|------|------|
| **feature_id** | 고유 feature 식별 번호 |
| **mz** | 관측된 m/z 값 (질량/전하) |
| **rt** | 머무름 시간 (초 단위) |
| **intensity** | 피크 강도 |
| **pfas_score** | **PFAS 우선순위 점수** (높을수록 PFAS 가능성↑) |
| **evidence_count** | 일치하는 증거 유형 수 |
| **evidence_types** | 일치한 증거 목록 (예: `mdc,kmd`) |

### 4.4 PFAS Score 산출 방식

| 증거 유형 | 가중치 | 설명 |
|----------|-------|------|
| `kmd_series` | +2.0 | CF2 동족체 시리즈 멤버 |
| `mdc_region` | +1.5 | PFAS MD/C 영역 내 위치 |
| `df_match` | +3.0 | MS2 진단 조각 이온 매칭 |
| `delta_m_match` | +2.5 | 중성 손실 규칙 매칭 (CF2, HF 등) |
| `suspect_match` | +5.0 | Suspect 리스트 매칭 |

**예시**: `pfas_score = 5.5` → `mdc_region(1.5) + kmd_series(2.0) + df_match(2.0)` 조합 가능

---

## 5. Visualization 탭

우선순위 지정된 Feature들을 시각적으로 탐색합니다.

### 5.1 시각화 유형

| View | 설명 |
|------|------|
| **EIC** | Extracted Ion Chromatogram - 선택한 m/z의 시간에 따른 intensity 변화 |
| **MS2 Spectrum** | 해당 feature의 MS/MS 단편화 스펙트럼 |
| **Correlations** | EIC 상관관계 네트워크 (동시 용리 화합물 탐지) |
| **Homologous Series** | 동족체 시리즈 정보 |

### 5.2 사용 방법
1. 좌측 테이블에서 feature 클릭
2. 상단 드롭다운에서 View 유형 선택
3. 우측에서 플롯 확인
4. **Export Plot**: 플롯 이미지 저장
5. **Export Viz Payload**: JSON 형식으로 데이터 내보내기

---

## 6. 결과 해석 방법

### 6.1 PFAS Score 해석 가이드

| 점수 범위 | 해석 | 후속 조치 |
|----------|------|----------|
| **≥ 7.0** | **매우 높음** - PFAS 가능성 높음 | 표준물질로 확인 권장 |
| **5.0~6.9** | **높음** - 강한 PFAS 의심 | MS2 스펙트럼 수동 검토 |
| **3.0~4.9** | **중간** - 추가 검토 필요 | 동족체 시리즈, EIC 패턴 확인 |
| **< 3.0** | **낮음** - PFAS 가능성 낮음 | 일반적으로 무시 가능 |

### 6.2 Evidence Types 해석

| 증거 | 의미 | 신뢰도 |
|------|------|-------|
| `suspect` | 알려진 PFAS DB와 m/z 일치 | ★★★★★ |
| `df_match` | CF3⁻, C2F5⁻ 등 진단 조각 검출 | ★★★★☆ |
| `kmd` | CF2 동족체 시리즈에 속함 | ★★★☆☆ |
| `mdc` | PFAS 특성 MD/C 영역에 위치 | ★★☆☆☆ |

### 6.3 결과 내보내기
- **Export Results** 버튼 → Excel (.xlsx) 또는 CSV 파일로 저장
- 저장된 파일에는 모든 스코어링 정보 포함

### 6.4 결과 보고 시 권장 사항
1. Score ≥ 5.0인 feature 목록 보고
2. 각 feature에 대해 evidence_types 명시
3. 가능하면 MS2 스펙트럼 첨부
4. 동족체 시리즈 패턴 그래프 포함

---

## 7. 문제 해결

### 7.1 일반적인 오류

| 오류 | 원인 | 해결 방법 |
|------|------|----------|
| "File not found" | mzML 경로 오류 | 파일 경로 확인, 한글 경로 피하기 |
| "Not centroided" | Profile 모드 데이터 | MSConvert로 centroid 변환 |
| "No features found" | 파라미터 문제 | Noise Threshold 낮추기, mzML 파일 확인 |
| "GPU not available" | CUDA 미설치 | Force CPU 모드 선택 |

### 7.2 성능 최적화 팁
- **대용량 파일**: GPU 모드 권장 (5000+ features)
- **메모리 부족**: Force CPU 모드, chunk size 조정
- **속도 향상**: Noise Threshold 높이기 (feature 수 감소)

### 7.3 권장 분석 순서
1. 기본 파라미터로 첫 분석 실행
2. Feature 수 확인 (100~1000개가 적정)
3. Feature 너무 많으면 → Noise Threshold ↑
4. Feature 너무 적으면 → Noise Threshold ↓, SNR ↓
5. Prioritization 결과 확인 후 Min Score 조정

---

## 부록: 용어 설명

| 용어 | 설명 |
|------|------|
| **Feature** | LC-MS에서 검출된 하나의 크로마토그래픽 피크 (m/z + RT + intensity) |
| **m/z** | 질량 대 전하비 (mass-to-charge ratio) |
| **RT** | Retention Time, 머무름 시간 |
| **KMD** | Kendrick Mass Defect, CF2 반복 단위 동족체 탐지에 사용 |
| **MD/C** | Mass Defect per Carbon, PFAS 특성 값 |
| **EIC** | Extracted Ion Chromatogram, 특정 m/z의 시간별 강도 변화 |
| **MS2** | 2차 질량 스펙트럼, 단편화 정보 |

---

**개발**: 김태형  
**연락처**: 010-9411-7143  
**(재)국제도시물정보과학연구원**
