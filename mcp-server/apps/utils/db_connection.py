import os
from supabase import create_client, Client


class DBConnection:
    """Supabase 데이터베이스 연결을 관리합니다."""

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL or SUPABASE_KEY is not set")

        self.supabase: Client = create_client(self.url, self.key)


if __name__ == "__main__":
    db_connection = DBConnection()
    print(db_connection.supabase.table("positions").select("*").execute())
