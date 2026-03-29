# Network-anomaly-detctor

A production-grade machine learning system for detecting anomalous network traffic, built on the CICIDS2017 dataset, a modern, realistic benchmark containing ~2.8 million labelled network flows across 79 features spanning both benign and attack traffic.

This project covers the full ML engineering lifecycle: exploratory data analysis, preprocessing and feature selection, model training and evaluation, production-ready inference API, containerisation, and data drift monitoring. The goal is to  demonstrate the practices that separate research notebooks from production systems such as structured code, proper error handling, meaningful version control, and reproducible pipelines.

A hybrid approach is used, combining traditional ensemble methods (XGBoost, Random Forest, Isolation Forest) with a deep learning baseline (TensorFlow/Keras), enabling direct comparison across paradigms for this class of problem.

## Dataset
CIC-IDS-2017 - Machine Learning CSV : https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=CIC-IDS-2017%2FCSVs 
The databse chosen 'CIDIDS2017' contains 2.8 million rows and is sperated across 8 files for different times of the working weeks representing 5 days of enwtork traffic with 14 different attack types and 79 feature.

## Project structure

## Setup
