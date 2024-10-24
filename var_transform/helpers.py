import os
from functools import wraps

def conditional_execution(env_var_name):
    """環境変数が 'true' の場合にのみ関数を実行するデコレーター"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 環境変数の値を取得
            enabled = os.getenv(env_var_name, 'false').lower() == 'true'
            if enabled:
                return func(*args, **kwargs)
            else:
                return None
        return wrapper
    return decorator
