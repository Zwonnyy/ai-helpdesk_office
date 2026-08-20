import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    sessionmaker,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATABASE_PATH = BASE_DIR / 'data' / 'helpdesk.db'

DATABASE_PATH = Path(
    os.getenv('HELPDESK_DATABASE_PATH', str(DEFAULT_DATABASE_PATH))
).expanduser().resolve()

DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


# Windows에서도 사용할 수 있도록
# C:\... -> C:/... 형태로 변환
DATABASE_URL = (
    f"sqlite:///"
    f"{DATABASE_PATH.as_posix()}"
)


# ============================================================
# BASE
# ============================================================

class Base(
    DeclarativeBase
):
    pass


# ============================================================
# ENGINE
# ============================================================

engine = create_engine(
    DATABASE_URL,

    connect_args={
        "check_same_thread": False,
    },

    pool_pre_ping=True,
)


# ============================================================
# SESSION FACTORY
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,

    autoflush=False,

    expire_on_commit=False,
)


# ============================================================
# DB DEPENDENCY
# ============================================================

def get_db() -> Generator[
    Session,
    None,
    None,
]:

    with SessionLocal() as session:
        yield session


# ============================================================
# INITIALIZE
# ============================================================

def init_db():

    # models를 import해야
    # SQLAlchemy metadata에 Ticket table이 등록됨
    from app import models  # noqa: F401

    Base.metadata.create_all(
        bind=engine
    )

    print(
        f"SQLite DB READY: "
        f"{DATABASE_PATH}"
    )
