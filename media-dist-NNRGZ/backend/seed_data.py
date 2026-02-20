# from database import SessionLocal
# from models import User, Device
# from datetime import datetime, timedelta
# import random

# db = SessionLocal()
# user = db.query(User).first() # Возьмем первого попавшегося юзера

# # Генерируем "историю успеха": с каждым днем устройств всё больше
# for day_offset in range(14, 0, -1):
#     new_devices_count = random.randint(1, 3) # Каждый день добавлялось 1-3 устройства
#     for _ in range(new_devices_count):
#         d = Device(
#             device_id=f"DEV-{random.randint(1000, 9999)}",
#             description="Тестовое устройство",
#             status="active",
#             user_id=user.id,
#             created_at=datetime.now() - timedelta(days=day_offset)
#         )
#         db.add(d)
# db.commit()
# print("Данные для прогноза загружены!")

from database import SessionLocal, engine
from models import Base, User, Device
from datetime import datetime, timedelta
import random
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def seed_downward_trend():
    db = SessionLocal()
    user = db.query(User).filter(User.username == "admin").first()
    db.query(Device).filter(Device.user_id == user.id).delete()

    # Имитируем 20 дней жизни системы
    total_pool = []
    for i in range(50): # Создаем пул из 50 устройств
        created_at = datetime.now() - timedelta(days=random.randint(10, 20))
        # Сначала они все активны
        total_pool.append({"id": f"DEV-{i}", "created": created_at, "status": "active"})

    # А теперь "отключаем" их со временем (чем ближе к сегодня, тем больше отключений)
    for i, dev in enumerate(total_pool):
        # Случайный день "смерти" устройства
        death_day = random.randint(1, 15)
        last_heartbeat = datetime.now() - timedelta(days=death_day)
        
        # Если "день смерти" уже прошел (относительно сегодня) - помечаем неактивным
        status = "blocked" if death_day > 2 else "active" # Большинство заблокированы
        
        new_dev = Device(
            device_id=dev["id"],
            description="Тест оттока",
            status=status,
            user_id=user.id,
            created_at=dev["created"],
            last_heartbeat=last_heartbeat
        )
        db.add(new_dev)
    
    db.commit()
    print("Данные с массовым отключением загружены!")

if __name__ == "__main__":
    seed_downward_trend()

# import math
# from database import SessionLocal, engine
# from models import Base, User, Device
# from datetime import datetime, timedelta
# import random
# from passlib.context import CryptContext

# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def seed_seasonal_data():
#     db = SessionLocal()
#     Base.metadata.create_all(bind=engine)

#     user = db.query(User).filter(User.username == "admin").first()
#     if not user:
#         user = User(
#             full_name="Администратор", username="admin",
#             hashed_password=pwd_context.hash("admin123"),
#             token="test-token-123", role="admin"
#         )
#         db.add(user)
#         db.commit()
#         db.refresh(user)

#     db.query(Device).filter(Device.user_id == user.id).delete()

#     print("Генерация циклической активности (сезонности)...")

#     # Генерируем данные за 30 дней
#     for day_offset in range(30, 0, -1):
#         # Создаем синусоиду: период 7 дней (неделя)
#         # math.sin принимает радианы, поэтому (day_offset * 2 * pi / 7)
#         cycle = math.sin(day_offset * (2 * math.pi / 7)) 
        
#         # Количество активных устройств будет колебаться от 10 до 30
#         target_active = int(1 + (cycle * 2)) 
        
#         for i in range(target_active):
#             # Часть устройств "активны", часть "заблокированы" для имитации динамики
#             # В пике цикла больше активных, в спаде - больше заблокированных
#             is_active = random.random() < (0.5 + (cycle * 0.4))
            
#             d = Device(
#                 device_id=f"DEV-{day_offset}-{i}",
#                 description="Циклический узел",
#                 status="active" if is_active else "blocked",
#                 user_id=user.id,
#                 created_at=datetime.now() - timedelta(days=day_offset),
#                 last_heartbeat=datetime.now() - timedelta(days=day_offset) if not is_active else datetime.now()
#             )
#             db.add(d)

#     db.commit()
#     print("Готово! Теперь на графике должна появиться 'волна'.")
#     db.close()

# if __name__ == "__main__":
#     seed_seasonal_data()
