# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.preprocessing import StandardScaler
# import datetime

# # Модель нейросети для оценки надежности устройства
# def train_reliability_model(devices_data):
#     """
#     Принимает список данных о девайсах.
#     Возвращает обученную модель и скалер.
#     """
#     if not devices_data:
#         return None, None

#     df = pd.DataFrame(devices_data)
    
#     # Признаки: кол-во файлов, время с последнего пинга (сек), частота ошибок (условно)
#     X = df[['files_count', 'seconds_since_heartbeat', 'error_rate']].values
#     # Цель: 1 - стабилен, 0 - склонен к сбою (генерируем синтетически для примера)
#     y = (X[:, 0] > 0) & (X[:, 1] < 120) 
#     y = y.astype(int)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = Sequential([
#         Dense(12, activation='relu', input_shape=(3,)),
#         Dense(8, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.fit(X_scaled, y, epochs=100, verbose=0)
    
#     return model, scaler

# def get_device_predictions(db_devices):
#     """
#     Формирует данные для BI-дашборда
#     """
#     prepared_data = []
#     current_time = datetime.datetime.now()

#     for d in db_devices:
#         files_count = len(d.files)
#         last_ping = d.last_heartbeat if d.last_heartbeat else current_time
#         seconds_since = (current_time - last_ping).total_seconds()
        
#         prepared_data.append({
#             'id': d.device_id,
#             'files_count': files_count,
#             'seconds_since_heartbeat': seconds_since,
#             'error_rate': 0.05 if d.status == 'active' else 0.8
#         })

#     if not prepared_data:
#         return []

#     model, scaler = train_reliability_model(prepared_data)
    
#     if model:
#         df = pd.DataFrame(prepared_data)
#         X_new = scaler.transform(df[['files_count', 'seconds_since_heartbeat', 'error_rate']].values)
#         predictions = model.predict(X_new)
        
#         for i, pred in enumerate(predictions):
#             val = float(pred[0])
#             # Математический трюк: возводим в степень < 1, чтобы подтянуть высокие значения к 1.0
#             # Например, 0.8 превратится в 0.93
#             optimized_score = np.power(val, 0.3) * 100
            
#             # Ограничиваем сверху 100%
#             prepared_data[i]['health_score'] = min(round(optimized_score, 1), 100.0)
    
#     return prepared_data

# import torch
# import torch.nn as nn
# import numpy as np
# from datetime import datetime

# # 1. Архитектура нейросети
# class DeviceHealthNN(nn.Module):
#     def __init__(self):
#         super(DeviceHealthNN, self).__init__()
#         # Вход: 2 признака. Скрытые слои: 8 -> 4. Выход: 3 класса (Ок, Warning, Critical)
#         self.network = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.ReLU(),
#             nn.Linear(8, 4),
#             nn.ReLU(),
#             nn.Linear(4, 3),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         return self.network(x)

# # Инициализация модели
# model = DeviceHealthNN()
# model.eval() # Переводим в режим предсказания

# def get_device_predictions(devices):
#     analytics_results = []
    
#     for dev in devices:
#         if not dev.last_heartbeat:
#             analytics_results.append({
#                 "id": dev.id,
#                 "device_id": dev.device_id,
#                 "health_score": 0,
#                 "status": "critical",
#                 "label": "Нет данных"
#             })
#             continue

#         # Подготовка данных (Preprocessing)
#         timeout = dev.user.heartbeat_timeout if dev.user else 60
#         diff = (datetime.now() - dev.last_heartbeat).total_seconds()
        
#         # Нормализация для нейросети
#         input_data = torch.tensor([[float(diff), float(diff/timeout)]], dtype=torch.float32)
        
#         # Получение предсказания (Inference)
#         with torch.no_grad():
#             prediction = model(input_data)
#             # prediction - это тензор вероятностей, например [[0.7, 0.2, 0.1]]
#             class_idx = torch.argmax(prediction).item()
#             health_score = float(prediction[0][0] * 100) # Вероятность класса "Healthy"

#         status_map = {0: "stable", 1: "warning", 2: "critical"}
#         label_map = {0: "Стабильно", 1: "Внимание", 2: "Критично"}

#         analytics_results.append({
#             "id": dev.id,
#             "device_id": dev.device_id,
#             "health_score": round(health_score, 1),
#             "status": status_map[class_idx],
#             "label": label_map[class_idx]
#         })
        
#     return analytics_results

import numpy as np
from sklearn.neural_network import MLPRegressor
from models import DeviceHistory
from datetime import timedelta

def moving_average(data, window_size=5):
    """Сглаживает данные методом скользящего среднего."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def get_device_predictions(db, user_id: int, days_to_predict: int = 7):
    history = db.query(DeviceHistory).filter(
        DeviceHistory.user_id == user_id
    ).order_by(DeviceHistory.timestamp.asc()).all()

    if len(history) < 20: # Увеличили порог для стабильности
        return {"error": "Недостаточно данных. Нужно накопить хотя бы 20 меток истории."}

    # Исходные данные
    start_time = history[0].timestamp
    X_raw = np.array([(h.timestamp - start_time).total_seconds() for h in history])
    y_raw = np.array([h.online_percentage for h in history])

    # --- УЛУЧШЕНИЕ 1: Сглаживание (Moving Average) ---
    window = 5
    y_smooth = moving_average(y_raw, window_size=window)
    # Сопоставляем X с укороченным после свертки массивом Y
    X_smooth = X_raw[window-1:].reshape(-1, 1) 

    # --- УЛУЧШЕНИЕ 2: Регуляризация (alpha) ---
    # alpha=1.0 — довольно сильная регуляризация, штрафующая за резкие изгибы
    nn_online = MLPRegressor(
        hidden_layer_sizes=(64, 32), 
        max_iter=3000, 
        alpha=0.01,           # Уменьшили с 1.0 до 0.01 для большей гибкости
        learning_rate_init=0.01,
        random_state=42
    )   
    
    nn_online.fit(X_smooth, y_smooth)

    # Прогноз
    last_time = history[-1].timestamp
    predictions = []
    
    for i in range(1, days_to_predict + 1):
        future_time = last_time + timedelta(days=i)
        future_seconds = np.array([[(future_time - start_time).total_seconds()]])
        
        pred_val = nn_online.predict(future_seconds)[0]
        predictions.append({
            "date": future_time.strftime("%d.%m"),
            "predicted_online": round(max(0.0, min(100.0, pred_val)), 1)
        })

    return {
        "historical": [
            {"date": h.timestamp.strftime("%d.%m %H:%M"), "online": round(h.online_percentage, 1)} 
            for h in history
        ],
        "predictions": predictions
    }