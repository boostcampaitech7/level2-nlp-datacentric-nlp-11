# Topic Classification - Data Centric

## 1 프로젝트 개요

**1.1 개요**

Topic Classification은 주어진 자연어 문장이 어떤 주제나 카테고리에 속하는지를 분류하는 NLP Task이다. 모델이 자연어의 주제를 얼마나 잘 이해하는지 평가할 수 있는 기본적인 방법 중 하나로, 특히 KLUE-Topic Classification benchmark에서는 뉴스 헤드라인을 보고 해당 뉴스가 어떤 주제를 가지는지 분류하는 작업이 요구된다.
본 대회는 데이터 품질 개선을 통해 모델의 성능을 높이는 Datacentric AI 접근법에 초점을 두고 있기 때문에 모델 개선은 할 수 없으며, 데이터셋을 정제할 때에도 여러 제한 사항을 준수하여야 한다. 특히, 실제 KLUE-Topic Classification benchmark에는 생활문화(Society), 스포츠(Sports), 세계(World), 정치(Politics), 경제(Economy), IT과학(IT/Science), 사회(Society)라는 주제가 공개되어 있으나, 본 대회에서는 이러한 라벨 매핑 정보는 사용할 수 없으며 오직 정수 인코딩 정보만 사용하여야 한다. 또한 외부 데이터셋(크롤링 포함)을 사용할 수 없고, 유료 결제가 필요한 비공개 생성형 모델도 사용할 수 없다.

<br />

**1.2 평가지표**

평가 지표는 Macro F1 Score이며, 이는 모든 라벨에 동등한 중요도를 부여하기 위함이다. Macro F1 Score는 라벨 별 f1 score의 평균으로 계산한다.

<img width="700" alt="평가지표" src="https://github.com/user-attachments/assets/c1c233a6-a67e-4b10-b368-fbe5b4f04ce4">

<br />

## 2 프로젝트 팀 구성 및 역할

## 팀원 소개

| **이름** | **프로필** | **역할** | **GitHub** |
| --- | --- | --- | --- |
| **강정완** | <img alt="강정완" width="140" height="140" src="https://github.com/user-attachments/assets/4f48f414-1da1-4476-acfa-b73104604db7" /> | - 텍스트 노이즈 클리닝 (by T5) <br /> - 라벨 오류 보정 (by Clustering) | [GJ98](https://github.com/GJ98) |
| **김민선** | <img alt="김민선" width="140" height="140" src="https://github.com/user-attachments/assets/603a2aaa-58ea-416e-b366-097f845bf5d5" /> | - 프로젝트 협업 관리(깃허브 이슈 템플릿 및 pre-commit 설정, <br /> Commitizen 설정, 노션 관리) <br /> - 노이즈 데이터 증강 (by LLM) <br /> - 데이터 증강 (by Back Translation) | [CLM-BONNY](https://github.com/CLM-BONNY) |
| **서선아** | <img alt="서선아" width="140" height="140" src="https://github.com/user-attachments/assets/57c9c737-28d7-4ed0-b8c9-48eb5daaeb8a" /> | - 텍스트 노이즈 클리닝 (by LLM) <br /> - 라벨 오류 보정 (by LLM) <br /> - 데이터 증강 (by LLM) | [seon03](https://github.com/seon03) |
| **이인구** | <img alt="이인구" width="140" height="140" src="https://github.com/user-attachments/assets/51a26c46-03e5-4a42-94de-9f7f3034383c" /> | - 데이터 증강(by Back Translation) | [inguelee](https://github.com/inguelee) |
| **이재협** | <img alt="이재협" width="140" height="140" src="https://github.com/user-attachments/assets/75b8ef71-6afe-4654-9843-a0913937a1de" /> | - 텍스트 노이즈 클리닝 (by LLM) <br /> - 데이터 증강 (by LLM) <br /> - 라벨 오류 보정 (by Cleanlab) | [jhyeop](https://github.com/jhyeop) |
| **임상엽** | <img alt="임상엽" width="140" height="140" src="https://github.com/user-attachments/assets/2d66dd95-8c1c-4441-9511-6bf2fc3b06170" /> | - 작업큐 구현 <br /> - 텍스트 노이즈 클리닝 (by LLM) <br /> - 라벨 오류 보정 (by LLM, Clustering) <br /> - 데이터 증강 (by LLM) | [gityeop](https://github.com/gityeop) |

<br />

## 3 프로젝트

**3.1 프로젝트 진행 일정**

- 각자 자신의 아이디어를 가지고 프로젝트 전반부터 후반까지 진행

<img width="700" alt="프로젝트 일정" src="https://github.com/user-attachments/assets/f251c79e-6e00-4b2c-ac06-3634ad71dc9d">

<br />
<br />

**3.2 프로젝트 폴더 구조**

```
├── data_augmentation/             # 데이터 증강 파일 포함 폴더
├── label_fixing/                  # 라벨 오류 보정 파일 포함 폴더
├── text_cleaning/                 # 노이즈 텍스트 복원 파일 포함 폴더
├── utils/                         # 유틸리티 함수 파일 포함 폴더 폴더
├── baseline_code.py               # 베이스라인 코드
```

<br />

## 4 EDA

**학습 데이터셋의 텍스트 노이즈(한글을 제외한 문자) 비율 분포**

<img width="450" src="https://github.com/user-attachments/assets/80142c7f-bfc5-4bb9-867c-f2f3521a6e40" />
    
<br />

## 5 프로젝트 수행

**5.0 Text Noise Classification**

- Rule-based Text Noise Classification
- LLM Text Noise Classification

**5.1 Text Cleaning**

- T5 Text Cleaning
- LLM Text Cleaning

**5.2 Label Fixing**

- CleanLab
- Clustering
- LLM Label Fixing

**5.3 Data Augmentation**

- Back Translation
- LLM Augmentation

<br />

## 6 Wrap-up Report

자세한 내용은 <a href="https://github.com/boostcampaitech7/level2-nlp-datacentric-nlp-11/blob/develop/Topic%20Classification%20%E1%84%83%E1%85%A2%E1%84%92%E1%85%AC_NLP_%E1%84%90%E1%85%B5%E1%86%B7%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3(11%E1%84%8C%E1%85%A9).pdf">**Wrap-up Report**</a>를 참고해 주세요 !
