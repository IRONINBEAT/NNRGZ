import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

# Архитектура: Простая нейросеть для прогноза временных рядов
class GrowthPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x): return self.net(x)

def predict_growth(devices, days_ahead=7):
    if len(devices) < 2:
        return {"history": [], "forecast": []}

    # 1. Подготовка данных: (день_от_старта -> кол-во устройств)
    start_date = min(d.created_at for d in devices)
    history = {}
    for d in devices:
        day = (d.created_at - start_date).days
        history[day] = history.get(day, 0) + 1
    
    # Накапливаем сумму (cumulative count)
    days = sorted(history.keys())
    counts = []
    current_total = 0
    for day in range(max(days) + 1):
        current_total += history.get(day, 0)
        counts.append(current_total)
    
    # 2. Обучение нейросети (быстрое, на лету)
    X = torch.tensor(range(len(counts)), dtype=torch.float32).view(-1, 1)
    y = torch.tensor(counts, dtype=torch.float32).view(-1, 1)
    
    model = GrowthPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for _ in range(200): # Обучаем 200 эпох
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    # 3. Прогноз
    future_days = torch.tensor(range(len(counts), len(counts) + days_ahead), dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        preds = model(future_days).flatten().tolist()

    return {
        "labels": [(start_date + timedelta(days=i)).strftime("%d.%m") for i in range(len(counts) + days_ahead)],
        "history": counts,
        "forecast": [None]*(len(counts)-1) + [counts[-1]] + [round(p, 1) for p in preds]
    }