import json

import redis

from app.config import settings


class ConversationMemory:
    def __init__(self) -> None:
        self.client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    def get_history(self, session_id: str) -> list[dict]:
        key = f"chat:{session_id}"

        messages = self.client.lrange(key, 0, -1)

        return [json.loads(message) for message in messages]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        key = f"chat:{session_id}"

        message = {
            "role": role,
            "content": content,
        }

        self.client.rpush(
            key,
            json.dumps(message),
        )

    def clear_history(self, session_id: str) -> None:
        key = f"chat:{session_id}"
        self.client.delete(key)