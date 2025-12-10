from sqlalchemy import create_engine, Integer, String, Column, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


username = "postgres"
password = 8212
host = "localhost"
port = 5432
database = "postgres"

db_url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(db_url)
Base = declarative_base()

class Table(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True)
    type = Column(String)
    vehicle_index = Column(Integer)
    action = Column(String)
    detected_at = Column(DateTime, default=func.now())

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

