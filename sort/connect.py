from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from tools import get_password
from datetime import datetime

username = "postgres"
password = get_password()
host = "localhost"
port = 5432
database_name = "bcs"

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
    status_at_moment = Column(String, nullable=False)
    total_at_moment = Column(Integer, nullable=False)
    time_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)




def get_data(session, tracked_obj_idx, type, total_at_moment, status_at_moment = "detected"):
    tracked_obj_idx = int(tracked_obj_idx)
    obj = (session.query(Vehicles).filter_by(tracked_obj_idx = tracked_obj_idx)
           .order_by(Vehicles.time_updated).first())

    if obj:
        if type is not None:
            obj.type = type
        if total_at_moment is not None:
            obj.total_at_moment = total_at_moment

        obj.time_updated = datetime.now()
        session.commit()

    else:
        t_obj = Vehicles(tracked_obj_idx = tracked_obj_idx, type = type, status_at_moment = status_at_moment, total_at_moment = total_at_moment)
        session.add(t_obj)
        session.commit()



def get_status(session, idx_list):
    # choose all tracked id
    obj = session.query(Vehicles.tracked_obj_idx).all()
    # go through them all (putting into int)
    for t_obj in obj:
        t_obj = int(t_obj.tracked_obj_idx)
        # if idx not in list of detected indices
        if t_obj not in idx_list:
            # then choose first string from table with the same id
            object_to_delete = session.query(Vehicles).filter_by(tracked_obj_idx=t_obj).first()
            # ana mark as not-detected one
            object_to_delete.status_at_moment = 'not_detected'
            session.commit()
        else:
            # otherwise, make sure that they're marked correctly
            object_to_repair = session.query(Vehicles).filter_by(tracked_obj_idx=t_obj).first()
            object_to_repair.status_at_moment = 'detected'
            session.commit()
