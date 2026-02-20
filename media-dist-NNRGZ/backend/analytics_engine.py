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
    if not devices: return None

    start_date = min(d.created_at for d in devices)
    total_days = (datetime.now() - start_date).days + 1
    
    # Считаем количество активных на каждый день i
    daily_active_count = []
    for i in range(total_days):
        check_date = start_date + timedelta(days=i)
        # Устройство активно, если оно уже создано И (оно активно ИЛИ оно "умерло" позже даты проверки)
        count = 0
        for d in devices:
            if d.created_at <= check_date:
                # Если устройство сейчас blocked, смотрим, когда был последний heartbeat
                if d.status == "active":
                    count += 1
                elif d.last_heartbeat and d.last_heartbeat > check_date:
                    count += 1
        daily_active_count.append(count)

    # Обучаем нейросеть на этом "падении"
    max_val = max(daily_active_count) if daily_active_count else 1
    X = torch.linspace(0, 1, steps=total_days).view(-1, 1)
    y = torch.tensor(daily_active_count, dtype=torch.float32).view(-1, 1) / max_val
    
    model = GrowthPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(600):
        optimizer.zero_grad()
        loss = torch.nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()

    # Прогноз
    future_x = torch.linspace(1, 1.4, steps=days_ahead).view(-1, 1)
    with torch.no_grad():
        preds = [max(0, p * max_val) for p in model(future_x).flatten().tolist()]

    return {
        "labels": [(start_date + timedelta(days=i)).strftime("%d.%m") for i in range(total_days + days_ahead)],
        "history": daily_active_count,
        "forecast": [None]*(total_days-1) + [daily_active_count[-1]] + preds,
        "metrics": {
            "avg_growth": round(daily_active_count[-1] - daily_active_count[0], 1),
            "server_load": round((daily_active_count[-1]/50)*100, 1),
            "days_monitored": total_days,
            "trend": "down" if preds[-1] < daily_active_count[-1] else "up"
        }
    }