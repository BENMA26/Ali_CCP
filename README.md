# Ali_CCP: Alibaba Click and Conversion Prediction

## Description

A deep learning pipeline for **Click-Through Rate (CTR)** and **Conversion Rate (CVR)** prediction on the [Alibaba CCP dataset](https://tianchi.aliyun.com/dataset/408), built with PyTorch.

This project implements a **two-tower retrieval architecture** with multi-task learning, covering the full recommendation pipeline from candidate retrieval to ranking:

- 🔍 **Recall Stage** — Dual-encoder two-tower model with negative sampling (random, popularity-weighted, in-batch)
- 🎯 **Ranking Stage** — Multi-task learning for joint CTR & CVR prediction
- 📦 **Sequence Modeling** — User behavior sequence preprocessing and encoding
- ⚡ **Training** — Supports pointwise / pairwise / listwise loss with early stopping and TensorBoard logging

## Installation

## Dataset