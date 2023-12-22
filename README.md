PatchTST_project
=============================
PatchTST 분석 및 개선을 위한 다양한 방법론   


PatchTST는 Transformer 모델이 Long-Term Time Series Forecasting(LTSF)에 적합하지 않다는 주장에 반박하기 위하여 연구된 모델이다. (Are Transformers effective for time series forecasting?, 2022) 이는 Transformer를 기반으로 설계된 다변량 시계열 예측 모델로서, 다음 두 가지 핵심 구성 요소에 기반한다. 

1] 시계열 데이터를 서브 시리즈 수준의 Patch로 분할하여 Transformer의 입력 토큰으로 사용
2] Channel-Independence를 통해 데이터의 각 채널을 단변량 시계열로 분해하여 학습

이 두 가지 특성을 통해, PatchTST는 Transformer Encoder에 들어가는 임베딩의 Locality가 유지되고, Look-back Window의 크기에 따른 Attention Map의 복잡도가 감소하며, 모델이 더 긴 History를 학습할 수 있다는 장점을 얻을 수 있다.

본 논문에서는 PatchTST의 기본적 원리를 탐구하고, Moving Average Decomposition, Non-stationary Method, FTT Decomposition과 같은 개선 방안을 제안하며, 해당 방식으로 실험을 진행한 결과와 분석을 제시한다.



# 1. Introduction
Time Series Analysis에 있어서 Time Series Forecasting은 시계열 분석에서 가장 주요한 연구 주제로, 현재까지도 활발히 연구되고 있는Supervised Learning이다.    
특히 트랜스포머 기반 딥러닝 모델의 빠른 성장과 함께 시계열 분석을 통한 예측에 대한 연구 작업이 크게 증가했다. (Bryan & Stefan 2021; Torres et al. 2021; Lara-Benítez et al. 2021). Attention을 기반으로 학습하는 트랜스포머는 Time Series Forecasting 뿐만 아니라 Input Data로부터 Feature Extraction을 수행하는 Representation Learning에서도 뛰어난 성능을 보여주었다.    이러한 트랜스포머는 Autoformer, Informr 등의 시계열 데이터 분석을 타겟으로 하는 다양한 Time Series Transformer(TST)들의 기초가 되었고,    
뛰어난 Representation Learning 성능을 Classification 및 Anomaly Detection과 같은 다양한 분야에 적용하며 높은 성능을 달성해왔다.    

 Transformer (Vaswani et al. 2017)는 자연어 처리(NLP) (Kalyan et al. 2021), 컴퓨터 비전(CV) (Khan et al. 2021), 음성 (Karita et al. 2019) 그리고 최근에는 시계열 (Wen et al. 2022)과 같은 다양한 응용 분야에서 큰 성공을 거두었다. 이는 시퀀스 내 요소 간의 연결 관계를 자동으로 학습할 수 있는 Attention메커니즘을 중심으로 하여 Sequential한 특징을 가지는 시계열 데이터의 모델링 작업에 이상적이다. 특히Informer (Zhou et al. 2021), Autoformer (Wu et al. 2021), FEDformer (Zhou et al. 2022)는 시계열 데이터에 성공적으로 적용된 Time Series Transformer(TST) 모델 중 일부이다. 그러나 Attention Mechanism을 위한 TST의 복잡한 설계에도 불구하고, 최근 논문 (Zeng et al. 2022)에서는 Linear Layer로 구성된 매우 단순한 다중 레이어 퍼셉트론(MLP)이 다양한 벤치마크에서 TST 모델들보다 낮은 복잡도와 높은 성능을 보이고 있다. 특히 장기 시계열 예측(Long Time Series Forecasting, LTSF) 분야에서 Transformer의 복잡도 대비 성능에 심각한 의문을 던지고 있다.    본 논문의 기반 연구에서 소개된 모델인 PatchTST는Patch-based Channel-independence Transformer (PatchTST) 모델이며, 이러한 Transformer의 유용성 문제에 대해 정면으로 맞서 LSTF 분야의 Transformer 성능을 보인 대표적인 연구이다. 본 논문에서는 LTSF에서 PatchTST의 Patching과 Channel-independence라는 두 가지 핵심 개념을 설명하며 성능 개선을 위한 다양한 방법론을 시도해 보고, Time Series Forecasting의 TST모델이 좋은 성능을 가지기 위한 요소를 분석하고자 한다.

## 1.1 Patching
>   Patching은 입력 시계열 데이터 시퀀스를 작은 서브 시퀀스들의 묶음으로 나누는 단계이다. 여기서 묶음 단위를 ‘Patch’라고 하며, 이는 PatchTST가 다른 TST들과 차별점을 가지게 되는 핵심적인 방법이다. Transformer는 Point-Wise Input Token 방식으로, Time point 별 시계열 데이터 각각을 입력 토큰으로 임베딩하여 한번에 처리한다. 이러한 방식은 개별 Time point로는 별다른 의미를 갖지 못하는 시계열 데이터의 연속적인 특성을 전혀 고려하지 못한다. Patching은 이와 반대로 Sequence-Wise Input Token 방식을 사용한다. 입력되는 시계열 시퀀스를 Look Back Window를 슬라이드하며 Window에 존재하는 데이터들의 토큰을 임베딩하는 방식으로 운영되기 때문에 모든 Time point를 토큰으로 임베딩하는 일반적인 Transformer보다 토큰의 개수가 현저히 줄어들게 된다.
 전처리를 통해 만들어진 Patch가 Transformer의 Encoder에 Token으로 입력되며, 이를 통해 시계열 데이터의 Attention 메커니즘에 의한 계산 복잡도를 개선하고, 모델이 더 많은 과거의 데이터를 학습할 수 있도록 한다.

## 1.2 Channel Independance
> CI 전략(Channel Independance)은 단변량 시계열 데이터를 미래 시계열 값으로 매핑하는 함수를 식별하는 방법이다. CD 전략(Channel Dependance)은 다변량 시계열 데이터를 미래 시계열 값으로 매핑한다. 현재 대부분의 SOTA(Sate-of-the-Art) LTSF 모델들은 위와 같은 CI 전략을 채택하고 있으며 이번에 다룬 PatchTST 모델도 채널 독립 방식을 따르고 있다. PatchTST 뿐만 아니라 DLinear등 현재 강력한 시계열 예측 모델은 모두 다변량 시계열 데이터를 분리하여 단변량(Channel Indepence)으로 예측하는데 사용하고 있다. 직관적으로 MTS(Multivariate Time Series)의 모든 과거 변수를 사용하여 동시에 모든 미래 변수를 예측하는 것이 적합해 보일 수 있다. CD방식은 CI방식의 단변량처리와 다르게 변수 간의 상호 관계를 포착하기 때문이다. 그러나 최근 연구에 따르면 채널 독립(CI) 전략이 채널 연관(CD)접근법을 능가한다는 것이 입증되었다.(The Capacity and Robustness Trade-off: Revisiting the ChannelIndependent Strategy for Multivariate Time Series Forecasting. 2023). 해당 논문에서는 다변량 데이터의 접근법에 상관관계인 ACF-Value를 도입하여 왜 CI 방법이 Distribution Shift에 강건한지 설명하고 있다.

## 1.3 PatchTST Architecture
<img src="/image/image_1.png" width="80%" height="40%" alt="참고이미지"></img>    

> 그림 (a)는 PatchTST를 구성하는 전체 아키텍처에 대한 그림이다. 초기에 Multivariate Input Sequence를 Channel-Independence하게 분리하여 Univariate로 만들고, 이를 Transformer Backbone에 입력한다. 모델은 Backbone내에서 각각의 Univarite를 Patching하고, Transformer Encoder를 거치며 데이터를 학습한다. 이를 통해 출력된 Output은 Look-back Window Size (L)만큼을 학습하여 예측된 미래의 T만큼의 결과이다.

# 2. Related Work
<img src="/image/image_2.png" width="80%" height="40%" alt="참고이미지"></img>   

최근 LTSF를 포함한 시계열 데이터 문제의 주요한 모델에는 강력한 트랜스포머가 장악하고 있다. Vanila Transformer, Autoformer, Informer등 트랜스포머 모델에 근거한 다양한 아종들이 나오며 NLP 분야를 넘어 다양한 분야에서 강력한 성능을 자랑하고 있다. 하지만 트랜스포머의 근본적인 문제로, 수많은 어텐션에서 기인하는 높은 계산 복잡도와 모델 복잡도를 꼽을 수 있다. 이러한 문제점을 지적하듯이, 최근 놀라울 정도로 간단한 Multi Layer Perceptron(MLP) 기반의 모델들이 트랜스포머 기반의 모델들보다 성능과 효율성 면에서 더 좋은 성능을 보이고 있다.   
대표적으로 “Are Transformers Effective for Time Series Forecasting?, 2022” 에서는 입력 데이터에 Decomposition을 적용 후, 1 Layer Linear network만을 적용하여 예측을 수행하는 NTSF-linear 모델을 제안하였다.    

    
뿐만 아니라 “Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures, 2022” 는 Continuous/Interval Sampling을 통하여 입력 시퀀스에 대한 Local/Global Temporal Information을 추출하고, 각 Sampling의 결과를 Concatenation한 후 Information Exchange Block을 통해 예측을 진행하는 LightTS 모델을 제안하였다. 해당 논문들에서 제시한 모델들이 LTSF 분야에서 트랜스포머 기반 모델들의 성능을 능가하면서, LTSF에서 트랜스포머의 효과에 대한 의문이 제기되고 있다. 이에 본 논문에서 이용하는 PatchTST는 이러한 Light Weight MLP들의 반격에 다시 맞서며 기존 연구 결과와는 다르게 트랜스포머가 LTSF에 효과적이라는 것을 증명하고 있다.


# 3. Proposed Method
>  PatchTST의 성능 개선을 위해 시도했던 3가지의 방법에 대해 설명한다.
자세한 설명은 추후 블로그에 추가하겠음.

## 3.1 Decomposing Signal Using Moving Average 
- PatchTST 성능 개선을 위한 첫번째 방법으로, 이동평균을 이용한 시계열 분해 기법을 적용한 버전입니다.
- 원본 데이터에서 이동평균을 이용하여 Trend와 이를 뺀 잔차 데이터를 구합니다.
- 분해된 Trend 데이터와 잔차 데이터를 독립적인 두개의 모델에 입력해 학습시키고, 해당 출력값을 더해주어 최종 출력값을 완성시킵니다.

## 3.2 Non stationary Scailing
- PatchTST 성능 개선을 위한 두번째 방법으로, 입력 데이터의 비정상성 정보를 적용시켜 학습한 버전입니다.
- 입력데이터가 RevIn으로 Instance Normalization되고, 해당 평균과 표준편차 정보를 기억했다가 출력 레이어에서 de-Normalization을 수행합니다.   
- 이때 평균과 표준 편차를 별도로 기억했다가 Projector라는 다중 레이어 퍼셉트론을 추가하여    
  기존 트랜스포머의 멀티헤드 어텐션에서 적용된 스케일링 과정에 새로운 Re-Scaling을 적용하였습니다.

## 3.3 Fast Furier Transform(FFT) - Top k Decomposition
- PatchTST 성능 개선을 위한 세번째 방법으로, 입력 데이터의 주요 주파수 데이터와 그 잔차 데이터를 학습한 버전입니다.
- 원본 데이터를 FFT하여 주요 K개의 주파수만 살린 시그널을 구하고, 이를 원본 데이터에 빼주어 잔차 데이터를 제작합니다.
- 이렇게 원본 데이터에서 분해된 주요 주파수 데이터, 잔차 데이터를 독립적인 두개의 모델에 입력해 학습시키고,
  해당 출력값을 더해주어 최종 출력값을 완성시킵니다.
