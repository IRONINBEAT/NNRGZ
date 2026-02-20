import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta

class GrowthPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # Углубляем сеть для лучшей аппроксимации
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Линейный выход позволяет расти бесконечно
        )
    def forward(self, x): return self.net(x)

def get_enhanced_analytics(devices, days_ahead=7):
    if len(devices) < 2:
        return None

    # Подготовка временной шкалы
    start_date = min(d.created_at for d in devices)
    total_days = (datetime.now() - start_date).days + 1
    
    history_map = {}
    for d in devices:
        day_idx = (d.created_at - start_date).days
        history_map[day_idx] = history_map.get(day_idx, 0) + 1
    
    counts = []
    current_total = 0
    for i in range(total_days):
        current_total += history_map.get(i, 0)
        counts.append(current_total)

    # Обучение нейросети
    max_count = max(counts)
    X = torch.linspace(0, 1, steps=len(counts)).view(-1, 1)
    y = torch.tensor(counts, dtype=torch.float32).view(-1, 1) / max_count # Нормализуем Y к [0, 1]

    model = GrowthPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(500):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()

    # Прогноз на будущее
    future_steps = torch.linspace(1, 1.3, steps=days_ahead).view(-1, 1)
    with torch.no_grad():
        preds_norm = model(future_steps).flatten().tolist()
        preds = [p * max_count for p in preds_norm]

    # Расчет доп. метрик
    avg_growth = len(devices) / total_days
    # Условный лимит сервера - 50 устройств (для BI)
    server_load = (len(devices) / 50) * 100 

    return {
        "labels": [(start_date + timedelta(days=i)).strftime("%d.%m") for i in range(total_days + days_ahead)],
        "history": counts,
        "forecast": [None]*(total_days-1) + [counts[-1]] + [round(p, 1) for p in preds],
        "metrics": {
            "avg_growth": round(avg_growth, 2),
            "server_load": round(server_load, 1),
            "days_monitored": total_days
        }
    }