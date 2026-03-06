from __future__ import annotations

from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


_PROMPT = ChatPromptTemplate.from_template(
    "Dựa vào lịch sử trò chuyện và câu hỏi mới nhất của người dùng, "
    "hãy diễn đạt lại câu hỏi thành một câu hỏi độc lập (tiếng Việt) "
    "có thể hiểu được mà không cần lịch sử. "
    "Chỉ trả về câu hỏi, không giải thích.\n\n"
    "Lịch sử:\n{history}\n\n"
    "Câu hỏi mới nhất: {question}"
)


class Reflection:
    def __init__(self, llm: BaseChatModel):
        self._chain = _PROMPT | llm | StrOutputParser()

    def rewrite(self, chat_history: List[dict], max_items: int = 20) -> str:
        if not chat_history:
            return ""
        history = chat_history[-max_items:]
        last = history.pop()
        question = last.get("content") or ""
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history
        )
        return self._chain.invoke({"history": history_text, "question": question})
