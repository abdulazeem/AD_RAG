# database/chat_manager.py

from sqlalchemy import create_engine, text, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Optional
from config.settings import settings
import uuid

Base = declarative_base()

class ChatSession(Base):
    """Model for chat sessions."""
    __tablename__ = 'chat_sessions'

    id = Column(String, primary_key=True)
    title = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ChatMessage(Base):
    """Model for chat messages."""
    __tablename__ = 'chat_messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_session_id = Column(String, nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(Text)  # JSON string of sources
    cost = Column(Text)  # Store cost as text
    timestamp = Column(DateTime, default=datetime.utcnow)


class ChatManager:
    """Manager for chat history operations."""

    def __init__(self):
        self.engine = create_engine(settings.database.postgres_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_chat_session(self, first_query: str) -> str:
        """Create a new chat session with title from first query.

        Args:
            first_query: The first query in the chat

        Returns:
            The chat session ID
        """
        session = self.SessionLocal()
        try:
            # Generate unique ID
            chat_id = str(uuid.uuid4())

            # Create title from first 20 characters
            title = first_query[:20].strip()
            if len(first_query) > 20:
                title += "..."

            # Create new session
            chat_session = ChatSession(
                id=chat_id,
                title=title,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(chat_session)
            session.commit()

            return chat_id
        finally:
            session.close()

    def add_message(
        self,
        chat_session_id: str,
        role: str,
        content: str,
        sources: Optional[str] = None,
        cost: Optional[float] = None
    ):
        """Add a message to a chat session.

        Args:
            chat_session_id: The chat session ID
            role: 'user' or 'assistant'
            content: The message content
            sources: JSON string of sources (optional)
            cost: Cost in USD (optional)
        """
        session = self.SessionLocal()
        try:
            message = ChatMessage(
                chat_session_id=chat_session_id,
                role=role,
                content=content,
                sources=sources,
                cost=str(cost) if cost is not None else None,
                timestamp=datetime.utcnow()
            )
            session.add(message)

            # Update session's updated_at timestamp
            session.execute(
                text("UPDATE chat_sessions SET updated_at = :now WHERE id = :id"),
                {"now": datetime.utcnow(), "id": chat_session_id}
            )

            session.commit()
        finally:
            session.close()

    def get_all_chat_sessions(self) -> List[Dict]:
        """Get all chat sessions ordered by most recent.

        Returns:
            List of chat session dictionaries
        """
        session = self.SessionLocal()
        try:
            result = session.execute(
                text("""
                    SELECT id, title, created_at, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                """)
            )

            chats = []
            for row in result:
                chats.append({
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3]
                })
            return chats
        finally:
            session.close()

    def get_chat_messages(self, chat_session_id: str) -> List[Dict]:
        """Get all messages for a chat session.

        Args:
            chat_session_id: The chat session ID

        Returns:
            List of message dictionaries
        """
        session = self.SessionLocal()
        try:
            result = session.execute(
                text("""
                    SELECT role, content, sources, cost, timestamp
                    FROM chat_messages
                    WHERE chat_session_id = :chat_id
                    ORDER BY timestamp ASC
                """),
                {"chat_id": chat_session_id}
            )

            messages = []
            for row in result:
                message = {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[4]
                }
                if row[2]:  # sources
                    message["sources"] = row[2]
                if row[3]:  # cost
                    message["cost"] = float(row[3])
                messages.append(message)

            return messages
        finally:
            session.close()

    def delete_chat_session(self, chat_session_id: str):
        """Delete a chat session and all its messages.

        Args:
            chat_session_id: The chat session ID
        """
        session = self.SessionLocal()
        try:
            # Delete messages first
            session.execute(
                text("DELETE FROM chat_messages WHERE chat_session_id = :chat_id"),
                {"chat_id": chat_session_id}
            )

            # Delete session
            session.execute(
                text("DELETE FROM chat_sessions WHERE id = :chat_id"),
                {"chat_id": chat_session_id}
            )

            session.commit()
        finally:
            session.close()
