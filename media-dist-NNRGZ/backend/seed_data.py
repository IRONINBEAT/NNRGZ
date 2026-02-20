# from database import SessionLocal
# from models import Device, User
# import datetime
# import random

# def seed():
#     db = SessionLocal()
#     try:
#         user = db.query(User).filter(User.role == 'admin').first() or db.query(User).first()
#         if not user:
#             print("Ошибка: Нет пользователей.")
#             return

#         # Очистим старые тестовые устройства, чтобы не мешали
#         db.query(Device).filter(Device.device_id.like("AI-NODE-%")).delete(synchronize_session=False)

#         print(f"Генерация идеальной выборки для {user.username}...")

#         for i in range(50):
#             # 90% идеальных устройств
#             is_perfect = random.random() < 0.9
            
#             if is_perfect:
#                 status = "active"
#                 # Идеальный пинг: от 5 до 30 секунд назад
#                 last_ping = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(5, 30))
#                 desc = f"Идеальный узел #{i}"
#             else:
#                 status = "blocked"
#                 # Ужасный пинг: 24 часа назад
#                 last_ping = datetime.datetime.now() - datetime.timedelta(hours=24)
#                 desc = f"Мертвый узел #{i}"
            
#             new_dev = Device(
#                 device_id=f"AI-NODE-{i:03d}",
#                 description=desc,
#                 status=status,
#                 user_id=user.id,
#                 last_heartbeat=last_ping
#             )
#             db.add(new_dev)
        
#         db.commit()
#         print("✅ База наполнена 50 устройствами (90% идеальных).")

#     except Exception as e:
#         db.rollback()
#         print(f"❌ Ошибка: {e}")
#     finally:
#         db.close()

# if __name__ == "__main__":
#     seed()

# import uuid
# from datetime import datetime, timedelta
# import random
# from sqlalchemy.orm import Session
# from database import SessionLocal, engine
# from models import Base, User, Device, File
# from passlib.context import CryptContext

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def seed_data():
#     db = SessionLocal()
    
#     # 1. Очистка старых данных (опционально, если хочешь чистую базу)
#     # Base.metadata.drop_all(bind=engine)
#     Base.metadata.create_all(bind=engine)

#     # 2. Создаем тестового пользователя, если его нет
#     admin = db.query(User).filter(User.username == "admin").first()
#     if not admin:
#         admin = User(
#             full_name="Администратор Системы",
#             username="admin",
#             hashed_password=pwd_context.hash("admin123"),
#             token="test-token-123",
#             role="admin",
#             heartbeat_timeout=60
#         )
#         db.add(admin)
#         db.commit()
#         db.refresh(admin)

#     # 3. Создаем набор устройств с разными сценариями
#     scenarios = [
#         {"desc": "Экран в Холле", "offset": 5},      # Ок (5 сек назад)
#         {"desc": "Ресепшн", "offset": 10},          # Ок (10 сек назад)
#         {"desc": "Лифт 1", "offset": 45},           # Ок (но близко к порогу 60)
#         {"desc": "Лифт 2", "offset": 120},          # Warning (2 мин назад)
#         {"desc": "Кафетерий", "offset": 300},       # Warning (5 мин назад)
#         {"desc": "Паркинг А", "offset": 3600},      # Critical (час назад)
#         {"desc": "Паркинг Б", "offset": 86400},     # Critical (сутки назад)
#         {"desc": "Конференц-зал", "offset": 2},     # Ок
#         {"desc": "Склад", "offset": 500},           # Critical
#         {"desc": "Зона отдыха", "offset": 15},      # Ок
#     ]

#     print(f"Генерация {len(scenarios)} устройств для пользователя {admin.username}...")

#     for i, s in enumerate(scenarios):
#         # Проверяем, существует ли уже такое устройство
#         dev_id = f"DEV-00{i+1}"
#         existing_dev = db.query(Device).filter(Device.device_id == dev_id).first()
        
#         last_seen = datetime.now() - timedelta(seconds=s["offset"])
        
#         if existing_dev:
#             existing_dev.last_heartbeat = last_seen
#             existing_dev.status = "active"
#         else:
#             new_dev = Device(
#                 device_id=dev_id,
#                 description=s["desc"],
#                 status="active",
#                 user_id=admin.id,
#                 last_heartbeat=last_seen,
#                 token_synced=True
#             )
#             db.add(new_dev)

#     # 4. Добавим пару тестовых файлов, чтобы BI не был пустым
#     if not db.query(File).filter(File.user_id == admin.id).first():
#         for i in range(3):
#             new_file = File(
#                 file_id=str(uuid.uuid4())[:8],
#                 url=f"/uploads/videos/test_video_{i}.mp4",
#                 description=f"Рекламный ролик {i+1}",
#                 user_id=admin.id
#             )
#             db.add(new_file)

#     db.commit()
#     print("Готово! Теперь зайди в систему под admin / admin123 и открой /web/analytics")
#     db.close()

# if __name__ == "__main__":
#     seed_data()

import random
from datetime import datetime, timedelta
from database import SessionLocal, engine
import models

def seed_analytics(username: str):
    db = SessionLocal()
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        print("Пользователь не найден!")
        return

    db.query(models.DeviceHistory).filter(models.DeviceHistory.user_id == user.id).delete()

    now = datetime.now()
    # --- УЛУЧШЕНИЕ 3: Больше данных (1000 точек) ---
    total_points = 1000
    points_per_day = 24 # представим, что собираем данные раз в час
    
    print(f"Генерируем {total_points} точек данных для {username}...")

    for i in range(total_points):
        # Идем назад во времени от текущего момента
        point_time = now - timedelta(hours=(total_points - i))
        
        # Основной фон — стабильный онлайн 95%
        base_line = 95.0
        # Добавляем случайный шум (поломки)
        noise = random.uniform(-10, 5)
        
        # Имитируем реальную жизнь: иногда онлайн «проседает» на время
        if 400 < i < 450: # искусственная авария в прошлом
            base_line = 70.0

        record = models.DeviceHistory(
            user_id=user.id,
            timestamp=point_time,
            total_devices=50,
            online_percentage=max(0, min(100, base_line + noise))
        )
        db.add(record)
        
        if i % 200 == 0: db.flush() # сбрасываем в БД пачками

    db.commit()
    print("✅ База заполнена большим объемом данных.")

if __name__ == "__main__":
    seed_analytics("admin")