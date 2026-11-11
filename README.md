---
title: Machine Lifespan Predictor
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: Predict a machine's remaining useful life from sensor data
---

# Machine Lifespan Predictor

Enter a machine's sensor readings (or upload its sensor history) and a machine-learning model
estimates how many operating cycles it can still run before failure.

Model: gradient boosting trained on NASA C-MAPSS turbofan engine data (simulated).
Results show benchmark performance, not proven performance on real factory machines.
