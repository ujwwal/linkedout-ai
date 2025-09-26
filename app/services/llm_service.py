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
        r"update\s+the\s+hook",
    ]
    _CTA_CHANGE_PATTERNS = [
        r"change\s+the\s+cta",
        r"change\s+cta",
        r"switch\s+the\s+cta",
        r"switch\s+cta",
        r"use\s+a\s+different\s+cta",
        r"different\s+cta",
        r"another\s+cta",
        r"new\s+cta",
        r"replace\s+the\s+cta",
        r"replace\s+cta",
        r"swap\s+the\s+cta",
        r"update\s+the\s+cta",
    ]
    _FRAMEWORK_CHANGE_PATTERNS = [
        r"change\s+the\s+framework",
        r"change\s+framework",
        r"switch\s+the\s+framework",
        r"switch\s+framework",
        r"use\s+a\s+different\s+framework",
        r"different\s+framework",
        r"another\s+framework",
        r"new\s+framework",
        r"replace\s+the\s+framework",
        r"replace\s+framework",
        r"swap\s+the\s+framework",
        r"update\s+the\s+framework",
    ]

    _HOOK_CHANGE_REGEX = re.compile("|".join(_HOOK_CHANGE_PATTERNS), re.IGNORECASE)
    _CTA_CHANGE_REGEX = re.compile("|".join(_CTA_CHANGE_PATTERNS), re.IGNORECASE)
    _FRAMEWORK_CHANGE_REGEX = re.compile("|".join(_FRAMEWORK_CHANGE_PATTERNS), re.IGNORECASE)
    _ALL_CHANGE_REGEX = re.compile(
        "|".join(_HOOK_CHANGE_PATTERNS + _CTA_CHANGE_PATTERNS + _FRAMEWORK_CHANGE_PATTERNS),
        re.IGNORECASE,
    )

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
        self.client_last_framework: Dict[str, str] = {}
        self.client_last_cta: Dict[str, str] = {}
        self.client_last_topic: Dict[str, str] = {}

        self.hooks = self._load_hooks()
        self.frameworks = self._load_frameworks()
        self.ctas = self._load_ctas()

    @staticmethod
    def _normalise_phrase(value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        text = text.strip('"').strip()
        text = text.strip('“”').strip()
        return text

    def _load_phrases(self, file_path: Path, skip_keywords: Optional[List[str]] = None) -> List[str]:
        if not file_path.exists():
            print(f"Dataset not found at {file_path}.")
            return []

        skip_keywords = [keyword.lower() for keyword in (skip_keywords or [])]
        phrases: List[str] = []
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    text = self._normalise_phrase(raw_line)
                    if not text:
                        continue
                    lowered = text.lower()
                    if skip_keywords and any(keyword in lowered for keyword in skip_keywords):
                        continue
                    phrases.append(text)
        except Exception as exc:
            print(f"Failed to load phrases from {file_path}: {exc}")
            return []

        if not phrases:
            print(f"No usable entries found in {file_path}.")
        return phrases

    def _load_hooks(self) -> List[str]:
        phrases = self._load_phrases(Path(settings.HOOKS_CSV_PATH), skip_keywords=["hooks"])
        if not phrases:
            phrases = [
                "Here's a thought most professionals overlook…",
                "Quick insight that changed my approach this year…",
            ]
        return phrases

    def _load_frameworks(self) -> List[str]:
        return self._load_phrases(Path(settings.FRAMEWORKS_CSV_PATH), skip_keywords=["frameworks"])

    def _load_ctas(self) -> List[str]:
        return self._load_phrases(Path(settings.CTA_CSV_PATH), skip_keywords=["ctas"])

    def _normalize_client_id(self, client_id: str) -> str:
        return client_id or "default"

    def _get_client_memory(self, client_id: str) -> str:
        history = self.client_memory.get(client_id, [])
        if not history:
            return ""

        recent_history = history[-5:]
        formatted_history = ""
        for interaction in recent_history:
            topic = interaction.get("topic") or interaction.get("query")
            hook = interaction.get("hook")
            framework = interaction.get("framework")
            cta = interaction.get("cta")
            formatted_history += f"User request: {topic}\n"
            if hook:
                formatted_history += f"Hook used: {hook}\n"
            if framework:
                formatted_history += f"Framework used: {framework}\n"
            if cta:
                formatted_history += f"CTA used: {cta}\n"
            formatted_history += f"Generated post: {interaction['response']}\n\n"
        return formatted_history

    def _update_client_memory(
        self,
        client_id: str,
        original_query: str,
        topic: str,
        hook: str,
        framework: str,
        cta: str,
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
                "framework": framework,
                "cta": cta,
                "response": response,
            }
        )

        if len(self.client_memory[client_id]) > 20:
            self.client_memory[client_id] = self.client_memory[client_id][-20:]

        if topic:
            self.client_last_topic[client_id] = topic
        if hook:
            self.client_last_hook[client_id] = hook
        if framework:
            self.client_last_framework[client_id] = framework
        if cta:
            self.client_last_cta[client_id] = cta

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
        cleaned = self._ALL_CHANGE_REGEX.sub(" ", query)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _is_hook_change_request(self, query: str) -> bool:
        return bool(query and self._HOOK_CHANGE_REGEX.search(query))

    def _is_cta_change_request(self, query: str) -> bool:
        return bool(query and self._CTA_CHANGE_REGEX.search(query))

    def _is_framework_change_request(self, query: str) -> bool:
        return bool(query and self._FRAMEWORK_CHANGE_REGEX.search(query))

    def _resolve_topic(self, client_id: str, query: str, reuse_previous: bool) -> str:
        cleaned = self._clean_query(query)
        if cleaned:
            topic = cleaned
        elif reuse_previous:
            topic = self.client_last_topic.get(client_id, "").strip()
        else:
            topic = query.strip()

        if not topic:
            topic = "LinkedIn post for my audience"

        return topic

    def _select_from_list(self, items: List[str], previous: Optional[str], force_change: bool) -> str:
        if not items:
            return ""

        candidates = items
        if previous and len(items) > 1:
            candidates = [item for item in items if item != previous]
            if not candidates and force_change:
                candidates = items
        return random.choice(candidates)

    def _select_hook(self, client_id: str, force_change: bool = False) -> str:
        previous = self.client_last_hook.get(client_id)
        selected = self._select_from_list(self.hooks, previous, force_change)
        if selected:
            self.client_last_hook[client_id] = selected
        return selected

    def _select_framework(self, client_id: str, force_change: bool = False) -> str:
        previous = self.client_last_framework.get(client_id)
        selected = self._select_from_list(self.frameworks, previous, force_change)
        if selected:
            self.client_last_framework[client_id] = selected
        return selected

    def _select_cta(self, client_id: str, force_change: bool = False) -> str:
        previous = self.client_last_cta.get(client_id)
        selected = self._select_from_list(self.ctas, previous, force_change)
        if selected:
            self.client_last_cta[client_id] = selected
        return selected

    def _build_prompt(
        self,
        topic: str,
        client_id: str,
        hook: str,
        framework: str,
        cta: str,
        is_pro_user: bool,
        previous_hook: Optional[str],
        previous_framework: Optional[str],
        previous_cta: Optional[str],
        hook_change_requested: bool,
        framework_change_requested: bool,
        cta_change_requested: bool,
    ) -> List[Dict[str, str]]:
        client_history = self._get_client_memory(client_id)
        similar_posts = self._retrieve_similar_posts(topic)

        system_prompt = (
            """
            
            """
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
                full_context += "Ensure the new hook feels noticeably different from the previous version."

        if framework:
            full_context += (
                "\n\nFRAMEWORK TO FOLLOW:\n"
                f"{framework}\n"
                "Use this framework to shape the narrative (sections, sequencing, and transitions) while keeping the copy natural."
            )
            if previous_framework and previous_framework != framework:
                full_context += f"\nDo not fall back to the former framework: {previous_framework}."
            if framework_change_requested:
                full_context += "\nMake the change of framework obvious in structure and flow."

        if cta:
            full_context += (
                "\n\nCALL-TO-ACTION REQUIREMENT:\n"
                f"Close the post with a CTA inspired by this line (adjust wording to fit tone while keeping the intent intact):\n{cta}\n"
                "Place the CTA as the final sentence or paragraph."
            )
            if previous_cta and previous_cta != cta:
                full_context += f"\nAvoid reusing the previous CTA phrase: {previous_cta}."
            if cta_change_requested:
                full_context += "\nEnsure the CTA feels clearly different from the prior one."

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
                    "Please craft a polished LinkedIn post about the topic above while respecting the hook, framework, and CTA requirements."
                ),
            },
        ]

        return messages

    def generate_post(self, query: str, client_id: str, is_pro_user: bool = False) -> str:
        try:
            client_key = self._normalize_client_id(client_id)

            hook_change_requested = self._is_hook_change_request(query)
            framework_change_requested = self._is_framework_change_request(query)
            cta_change_requested = self._is_cta_change_request(query)

            reuse_previous_topic = hook_change_requested or framework_change_requested or cta_change_requested

            previous_hook = self.client_last_hook.get(client_key)
            previous_framework = self.client_last_framework.get(client_key)
            previous_cta = self.client_last_cta.get(client_key)

            topic = self._resolve_topic(client_key, query, reuse_previous_topic)
            selected_hook = self._select_hook(client_key, force_change=hook_change_requested)
            selected_framework = self._select_framework(client_key, force_change=framework_change_requested)
            selected_cta = self._select_cta(client_key, force_change=cta_change_requested)

            messages = self._build_prompt(
                topic=topic,
                client_id=client_key,
                hook=selected_hook,
                framework=selected_framework,
                cta=selected_cta,
                is_pro_user=is_pro_user,
                previous_hook=previous_hook,
                previous_framework=previous_framework,
                previous_cta=previous_cta,
                hook_change_requested=hook_change_requested,
                framework_change_requested=framework_change_requested,
                cta_change_requested=cta_change_requested,
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
                framework=selected_framework,
                cta=selected_cta,
                response=generated_text,
            )

            return generated_text

        except Exception as exc:
            error_message = f"Error generating post: {exc}"
            print(error_message)
            return "I'm sorry, I encountered an error while generating your LinkedIn post. Please try again later."
