PatchTST_project
=============================
PatchTST 분석 및 개선을 위한 다양한 방법론

### ABSTRACT
>  PatchTST는 Transformer 모델이 Long-Term Time Series Forecasting(LTSF)에 적합하지 않다는 주장에 반박하기 위하여 연구된 모델이다. (Are Transformers effective for time series forecasting?, 2022) 이는 Transformer를 기반으로 설계된 다변량 시계열 예측 모델로서, 다음 두 가지 핵심 구성 요소에 기반한다. 

1] 시계열 데이터를 서브 시리즈 수준의 Patch로 분할하여 Transformer의 입력 토큰으로 사용
2] Channel-Independence를 통해 데이터의 각 채널을 단변량 시계열로 분해하여 학습

이 두 가지 특성을 통해, PatchTST는 Transformer Encoder에 들어가는 임베딩의 Locality가 유지되고, Look-back Window의 크기에 따른 Attention Map의 복잡도가 감소하며, 모델이 더 긴 History를 학습할 수 있다는 장점을 얻을 수 있다.

본 논문에서는 PatchTST의 기본적 원리를 탐구하고, Moving Average Decomposition, Non-stationary Method, FTT Decomposition과 같은 개선 방안을 제안하며, 해당 방식으로 실험을 진행한 결과와 분석을 제시한다.



>  ###
