# 공장 예지보전 시스템

###### 2024.8 (1人 개인 프로젝트)

<br />

## 📌 Summary

**AI 를 이용한 공장 위험도 분석**

- 공장 센서를 통해 온도, 진동을 받아온 후 위험도를 분석.

<img width="1125" alt="스크린샷 2024-11-20 오전 10 49 30" src="https://github.com/user-attachments/assets/b60936ce-f1a6-4c4e-af3f-4536fbde1516">

<br />

## 🤔 Background

RS-485 유선 네트워크를 통해 온도와 진동 값을 수집하였다.
Telegraf를 사용하여 이 데이터를 InfluxDB에 저장하여 데이터베이스를 구축하였다.
이후 Grafana를 이용하여 데이터를 시각화하였다.
XGBoost 알고리즘을 활용하여 위험도를 분류하는 학습을 진행하였으며,
위험도는 안전, 주의, 경고, 위험의 4가지 등급으로 나누어 분류하였다.


<br />

## 🔨 Technology Stack(s)

Frontend : Grafana

Backend : Flask

DataBase : InfluxDB

<br />

## 🤩 Preview

![image (7)](https://github.com/user-attachments/assets/a4e569f9-a751-44e5-a429-a5763e361b93)





