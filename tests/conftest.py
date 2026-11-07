import os
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql

from vooy_gallery.store import Store


@pytest.fixture
def pg_store_factory():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests")
    schema = "test_gallery_" + uuid4().hex

    def factory(data_dir):
        return Store(data_dir, database_url=database_url, schema=schema)

    try:
        yield factory
    finally:
        with psycopg.connect(database_url, connect_timeout=5) as connection:
            connection.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
