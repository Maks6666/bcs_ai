from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from datetime import datetime

username = "postgres"
password = 134472
host = "localhost"
port = 5432
database_name = "vehicles"

# run in terminal: pip install psycopg2 OR pip install psycopg2-binary (if first doesnt work)
db_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database_name}"

try:
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    Base = declarative_base()
    print("Connected to DB")
except Exception as e:
    print(f"Connection failed: {e}")

class Vehicles(Base):
    __tablename__ = "vehicles"
    basic_id = Column(Integer, primary_key=True, index=True)
    tracked_obj_idx = Column(Integer, nullable=False)
    type = Column(String, nullable=False)
    total_at_moment = Column(Integer, nullable=False)
    time_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)




def get_data(session, tracked_obj_idx, type, total_at_moment, time_updated):
    obj = session.query(Vehicles).filter_by(tracked_obj_idx = tracked_obj_idx).order_by(time_updated.desc()).first()
    if obj:
        obj.type = type
        obj.total_at_moment = total_at_moment
        obj.time_updated = time_updated