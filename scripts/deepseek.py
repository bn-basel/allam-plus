"""DeepSeek V4.1 Flash through OpenRouter: the base model of our rule-driven data-generation pipeline (MIT licence;
its terms allow training on outputs).

The key is read from a local .env (this repo's, else the receptionist project's, which holds the same spend-capped
key); it is never printed or stored. Providers are pinned and data collection is denied, as in the earlier project.
Every call returns the cost OpenRouter reports, and Budget stops a job before it passes its cap.
"""

import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "deepseek/deepseek-v4.1-flash"
PROVIDERS = ["deepinfra", "fireworks"]
PRICE = (0.45, 1.80)  # USD per 1M tokens (in, out) at the dearest allowed provider: only used if no cost is reported


def load_key() -> str:
    for env in (ROOT / ".env", ROOT.parent / "saudi-dialect" / ".env"):
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("OPENROUTER_API_KEY", "")


class Budget:
    def __init__(self, cap_usd: float, ledger: Path):
        self.cap, self.ledger, self.lock = cap_usd, ledger, threading.Lock()
        self.spent = sum(json.loads(l)["usd"] for l in open(ledger, encoding="utf-8")) if ledger.exists() else 0.0

    def ok(self) -> bool:
        return self.spent < self.cap

    def add(self, usd: float, what: str) -> None:
        with self.lock:
            self.spent += usd
            with open(self.ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(time.time()), "usd": round(usd, 6), "what": what}) + "\n")


class DeepSeek:
    def __init__(self):
        import openai
        self.openai = openai
        self.client = openai.OpenAI(api_key=load_key(), base_url="https://openrouter.ai/api/v1", timeout=300)

    def chat(self, system: str, user: str, max_tokens: int = 6000, temperature: float = 0.9) -> dict:
        """One call, reasoning off (the output is the data, and thinking tokens cost money). Returns {"text", "usd"}
        or {"error"} without the error body."""
        try:
            r = self.client.chat.completions.create(
                model=MODEL, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body={"provider": {"order": PROVIDERS, "only": PROVIDERS, "data_collection": "deny"},
                            "reasoning": {"enabled": False}, "usage": {"include": True}})
        except self.openai.APIStatusError as e:
            return {"error": f"HTTP {e.status_code}"}
        except self.openai.APIConnectionError:
            return {"error": "connection error"}
        if not r.choices:
            return {"error": "empty response"}
        usage = r.usage
        extra = (usage.model_extra or {}) if usage else {}
        usd = extra.get("cost")
        if usd is None and usage:
            usd = (usage.prompt_tokens * PRICE[0] + usage.completion_tokens * PRICE[1]) / 1e6
        return {"text": r.choices[0].message.content or "", "usd": float(usd or 0.0),
                "tokens_in": usage.prompt_tokens if usage else 0, "tokens_out": usage.completion_tokens if usage else 0,
                "finish": r.choices[0].finish_reason}
