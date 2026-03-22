import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.models.schemas import ChatSummary, ChatHistoryResponse, ChatMessage

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)


def create_chat(title: str) -> str:
    """Creates a new chat history file and returns the chat_id."""
    chat_id = str(uuid.uuid4())
    chat_file = HISTORY_DIR / f"{chat_id}.json"
    
    chat_data = {
        "chat_id": chat_id,
        "title": title,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "messages": []
    }
    
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)
        
    return chat_id


def append_messages(chat_id: str, new_messages: list[ChatMessage]) -> None:
    """Appends new messages to an existing chat history."""
    chat_file = HISTORY_DIR / f"{chat_id}.json"
    if not chat_file.exists():
        raise FileNotFoundError(f"Chat {chat_id} not found.")
        
    with open(chat_file, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
        
    chat_data["messages"].extend([msg.model_dump() for msg in new_messages])
    chat_data["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    with open(chat_file, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)


def get_chat(chat_id: str) -> Optional[ChatHistoryResponse]:
    """Retrieves a specific chat history."""
    chat_file = HISTORY_DIR / f"{chat_id}.json"
    if not chat_file.exists():
        return None
        
    with open(chat_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return ChatHistoryResponse(**data)


def list_chats() -> list[ChatSummary]:
    """Lists all available chats, sorted by newest first."""
    chats = []
    for chat_file in HISTORY_DIR.glob("*.json"):
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                chats.append(ChatSummary(
                    chat_id=data["chat_id"],
                    title=data.get("title", "Unknown Chat"),
                    updated_at=data.get("updated_at", "")
                ))
        except (json.JSONDecodeError, KeyError):
            continue
            
    # Sort by updated_at descending
    return sorted(chats, key=lambda x: x.updated_at, reverse=True)


def delete_chat(chat_id: str) -> bool:
    """Deletes a chat history file."""
    chat_file = HISTORY_DIR / f"{chat_id}.json"
    if chat_file.exists():
        chat_file.unlink()
        return True
    return False
