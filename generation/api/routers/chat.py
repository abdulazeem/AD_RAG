# rag_app/generation/api/routers/chat.py

from fastapi import APIRouter
from typing import List
from generation.api.schemas import ChatSessionResponse, ChatMessageResponse
from database.chat_manager import ChatManager

router = APIRouter()
chat_manager = ChatManager()


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions():
    """Get all chat sessions."""
    sessions = chat_manager.get_all_chat_sessions()
    return [
        ChatSessionResponse(
            id=s["id"],
            title=s["title"],
            created_at=s["created_at"].isoformat(),
            updated_at=s["updated_at"].isoformat()
        )
        for s in sessions
    ]


@router.get("/sessions/{chat_session_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(chat_session_id: str):
    """Get all messages for a specific chat session."""
    messages = chat_manager.get_chat_messages(chat_session_id)
    return [
        ChatMessageResponse(
            role=m["role"],
            content=m["content"],
            sources=m.get("sources"),
            cost=None,  # Phoenix handles cost tracking — no need to store locally
            timestamp=m["timestamp"].isoformat()
        )
        for m in messages
    ]


@router.delete("/sessions/{chat_session_id}")
async def delete_chat_session(chat_session_id: str):
    """Delete a chat session and all its messages."""
    chat_manager.delete_chat_session(chat_session_id)
    return {"success": True, "message": "Chat session deleted"}
