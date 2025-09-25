import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import AzureOpenAI

from ..core.config import settings


class LLMService:
    """Service responsible for orchestrating LinkedIn post generation."""

    _HOOK_CHANGE_PATTERNS = [
        r"change\s+the\s+hook",
        r"change\s+hook",
        r"switch\s+the\s+hook",
        r"switch\s+hook",
        r"use\s+a\s+different\s+hook",
        r"different\s+hook",
        r"another\s+hook",
        r"new\s+hook",
        r"replace\s+the\s+hook",
        r"replace\s+hook",
        r"swap\s+the\s+hook",
    ]
    _HOOK_CHANGE_REGEX = re.compile("|".join(_HOOK_CHANGE_PATTERNS), re.IGNORECASE)
    _FILLER_REGEX = re.compile(r"\b(please|thanks|thank you)\b", re.IGNORECASE)

    def __init__(self, vector_store_service):
        self.client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.vector_store_service = vector_store_service

        self.client_memory: Dict[str, List[Dict]] = {}
        self.client_last_hook: Dict[str, str] = {}
        self.client_last_topic: Dict[str, str] = {}

        self.hooks = self._load_hooks()

    def _normalize_client_id(self, client_id: str) -> str:
        return client_id or "default"

    def _load_hooks(self) -> List[str]:
        hooks_path = Path(settings.HOOKS_CSV_PATH)
        if not hooks_path.exists():
            print(f"Hooks file not found at {hooks_path}. Hooks will be disabled.")
            return []

        hooks: List[str] = []
        try:
            with hooks_path.open("r", encoding="utf-8") as file:
                for raw_line in file:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue

                    if stripped.startswith('"""') and stripped.endswith('"""'):
                        hook = stripped.strip('"').strip()
                        if hook:
                            hooks.append(hook)
            if not hooks:
                print(f"No hooks extracted from {hooks_path}.")
        except Exception as exc:
            print(f"Failed to load hooks from {hooks_path}: {exc}")
        return hooks

    def _get_client_memory(self, client_id: str) -> str:
        history = self.client_memory.get(client_id, [])
        if not history:
            return ""

        recent_history = history[-5:]
        formatted_history = ""
        for interaction in recent_history:
            topic = interaction.get("topic") or interaction.get("query")
            hook = interaction.get("hook")
            formatted_history += f"User request: {topic}\n"
            if hook:
                formatted_history += f"Hook used: {hook}\n"
            formatted_history += f"Generated post: {interaction['response']}\n\n"
        return formatted_history

    def _update_client_memory(
        self,
        client_id: str,
        original_query: str,
        topic: str,
        hook: str,
        response: str,
    ) -> None:
        if client_id not in self.client_memory:
            self.client_memory[client_id] = []

        self.client_memory[client_id].append(
            {
                "timestamp": time.time(),
                "query": original_query,
                "topic": topic,
                "hook": hook,
                "response": response,
            }
        )

        if len(self.client_memory[client_id]) > 20:
            self.client_memory[client_id] = self.client_memory[client_id][-20:]

        if topic:
            self.client_last_topic[client_id] = topic
        if hook:
            self.client_last_hook[client_id] = hook

    def _retrieve_similar_posts(self, topic: str, top_k: int = 3) -> str:
        try:
            similar_docs = self.vector_store_service.search_similar_posts(topic, k=top_k)
            if not similar_docs:
                return ""

            examples = "Here are some example LinkedIn posts that might be relevant:\n\n"
            for index, doc in enumerate(similar_docs, 1):
                metadata = getattr(doc, "metadata", {}) or {}
                author = metadata.get("profile_name") or "Unknown Author"
                post_date = metadata.get("post_date") or ""
                profile_url = metadata.get("profile_url") or ""

                header_parts = [f"Example {index}"]
                if author:
                    header_parts.append(f"by {author}")
                if post_date:
                    header_parts.append(f"({post_date})")

                header = " ".join(header_parts)
                examples += f"{header}:\n{doc.page_content.strip()}\n"
                if profile_url:
                    examples += f"Source: {profile_url}\n"
                examples += "\n"
            return examples
        except Exception as exc:
            print(f"Error retrieving similar posts: {exc}")
            return ""

    def _clean_query(self, query: str) -> str:
        cleaned = query
        for pattern in self._HOOK_CHANGE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = self._FILLER_REGEX.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _is_hook_change_request(self, query: str) -> bool:
        if not query:
            return False
        return bool(self._HOOK_CHANGE_REGEX.search(query))

    def _resolve_topic(self, client_id: str, query: str, is_hook_change: bool) -> str:
        cleaned = self._clean_query(query)
        if cleaned:
            topic = cleaned
        else:
            topic = self.client_last_topic.get(client_id, "").strip()

        if not topic:
            topic = query.strip()

        if not topic:
            topic = "LinkedIn post for my audience"

        return topic

    def _select_hook(self, client_id: str, force_change: bool = False) -> str:
        if not self.hooks:
            return ""

        previous_hook = self.client_last_hook.get(client_id)
        candidate_hooks = self.hooks

        if previous_hook and len(self.hooks) > 1:
            candidate_hooks = [hook for hook in self.hooks if hook != previous_hook]

        if force_change and previous_hook and not candidate_hooks:
            candidate_hooks = self.hooks

        selected = random.choice(candidate_hooks) if candidate_hooks else random.choice(self.hooks)
        self.client_last_hook[client_id] = selected
        return selected

    def _build_prompt(
        self,
        topic: str,
        client_id: str,
        hook: str,
        is_pro_user: bool,
        previous_hook: Optional[str],
        hook_change_requested: bool,
    ) -> List[Dict[str, str]]:
        client_history = self._get_client_memory(client_id)
        similar_posts = self._retrieve_similar_posts(topic)

        system_prompt = (
            "You are LinkedOut, an AI assistant specialized in creating engaging LinkedIn posts.\n"
            "Follow these guidelines:\n"
            "- Write in a professional but conversational tone\n"
            "- Include relevant hashtags at the end\n"
            "- Keep posts concise (under 1300 characters)\n"
            "- Focus on providing value to the reader\n"
            "- Avoid clichés and generic corporate language\n"
            "- Format with appropriate line breaks and tasteful emoji where natural"
        )

        if hook:
            system_prompt += (
                "\n- Start the post with the provided hook on its own line (customise bracketed placeholders to match the topic)"
            )

        if is_pro_user:
            system_prompt += (
                "\n- For PRO users: Include more sophisticated content structures"
                "\n- For PRO users: Add a hook at the beginning and call-to-action at the end"
                "\n- For PRO users: Optimize for maximum engagement with advanced storytelling techniques"
            )

        full_context = system_prompt

        if hook:
            full_context += (
                "\n\nHOOK REQUIREMENTS:\n"
                f"Use this hook as the opening line (adapt placeholders like [industry] or [goal] to the topic):\n{hook}\n"
            )
            if previous_hook and previous_hook != hook:
                full_context += f"Avoid reusing the previous hook: {previous_hook}\n"
            if hook_change_requested:
                full_context += (
                    "Ensure the new hook feels noticeably different from the previous version while keeping the overall message aligned."
                )

        full_context += f"\n\nTOPIC OR REQUEST:\n{topic}"

        if client_history:
            full_context += "\n\nPREVIOUS INTERACTIONS WITH THIS USER:\n" + client_history

        if similar_posts:
            full_context += "\n\n" + similar_posts

        messages = [
            {"role": "system", "content": full_context},
            {
                "role": "user",
                "content": (
                    "Please craft a polished LinkedIn post about the topic above while respecting the hook requirements."
                ),
            },
        ]

        return messages

    def generate_post(self, query: str, client_id: str, is_pro_user: bool = False) -> str:
        try:
            client_key = self._normalize_client_id(client_id)
            hook_change_requested = self._is_hook_change_request(query)
            previous_hook = self.client_last_hook.get(client_key)

            topic = self._resolve_topic(client_key, query, hook_change_requested)
            selected_hook = self._select_hook(client_key, force_change=hook_change_requested)

            messages = self._build_prompt(
                topic=topic,
                client_id=client_key,
                hook=selected_hook,
                is_pro_user=is_pro_user,
                previous_hook=previous_hook,
                hook_change_requested=hook_change_requested,
            )

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0.5,
                presence_penalty=0.5,
            )

            generated_text = response.choices[0].message.content
            if generated_text is None:
                generated_text = "I couldn't generate a LinkedIn post at this time. Please try again."

            self._update_client_memory(
                client_id=client_key,
                original_query=query,
                topic=topic,
                hook=selected_hook,
                response=generated_text,
            )

            return generated_text

        except Exception as exc:
            error_message = f"Error generating post: {exc}"
            print(error_message)
            return "I'm sorry, I encountered an error while generating your LinkedIn post. Please try again later."