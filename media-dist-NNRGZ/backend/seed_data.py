from database import SessionLocal
from models import User, Device
from datetime import datetime, timedelta
import random

db = SessionLocal()
user = db.query(User).first() # Возьмем первого попавшегося юзера

# Генерируем "историю успеха": с каждым днем устройств всё больше
for day_offset in range(14, 0, -1):
    new_devices_count = random.randint(1, 3) # Каждый день добавлялось 1-3 устройства
    for _ in range(new_devices_count):
        d = Device(
            device_id=f"DEV-{random.randint(1000, 9999)}",
            description="Тестовое устройство",
            status="active",
            user_id=user.id,
            created_at=datetime.now() - timedelta(days=day_offset)
        )
        db.add(d)
db.commit()
print("Данные для прогноза загружены!")