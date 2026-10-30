"""Stage 5: model adapters. Each one turns (system prompt, messages, tools) into one step:
    {"text": str | None, "tool_calls": [{"name": ..., "arguments": {...}}], "raw": str}

  oracle           Always answers with the case's first expected option. Only for testing
                   the harness: it must score 100%.
  hf:<repo_id>     A local Hugging Face model in 4-bit (the same way it will be trained),
                   greedy decoding so every run gives the same output.
  llamacpp:<url>@<repo_id>  A GGUF build served by llama.cpp's llama-server at <url>. The prompt is rendered
                   by <repo_id>'s tokenizer exactly as for hf: and sent as token ids, so a re-test measures
                   only the GGUF file (merged weights, 8- or 4-bit), not a different prompt. Greedy decoding.
  llamachat:<url>@<repo_id>  The same llama-server, used the way apps use it (LM Studio, OpenAI-style clients):
                   its chat API builds the prompt with the GGUF's own template (start llama-server with
                   --jinja) and parses Gemma's tool calls itself. Greedy decoding.
  ollama:<model>@<repo_id>  A model imported into a local Ollama (http://127.0.0.1:11434), used the way Ollama's
                   users use it: through its chat API, with Ollama's own template and tool-call parsing (for
                   ALLaM, the plain-text tool format of release/allam/allam_tools.py). <repo_id> only decides
                   which of the two formats applies. Greedy decoding.
  or:<competitor>  An API model through OpenRouter, one of COMPETITORS below: a fixed model,
                   fixed providers (so results are reproducible), thinking on, max_tokens 2048
                   (the same cap the production setup uses), providers that store data excluded.

The OpenRouter key comes from the environment or a local .env file (OPENROUTER_API_KEY).
Keys, request headers and error text are never printed or stored: a failed call returns
{"error": "HTTP <status>"} and is not saved as a prediction.
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_MAX_TOKENS = 2048  # thinking included

# Each competitor: OpenRouter model id, the providers allowed (tried in this order), and
# USD per 1M tokens (input, output) at the most expensive allowed provider, used only as an
# estimate when OpenRouter does not report the real cost.
COMPETITORS = {
    # The model Mudeef runs on in production: the "beat what I have" comparison.
    "deepseek-v4.1-flash": {"model": "deepseek/deepseek-v4.1-flash", "providers": ["deepinfra", "fireworks"],
                            "price": (0.45, 1.80)},
    # One established model from each other major vendor, on the cheaper "flex" tier when available.
    "gpt-5.4-mini": {"model": "openai/gpt-5.4-mini", "providers": ["openai/flex", "openai"], "price": (0.75, 4.50)},
    "gemini-3.8-flash": {"model": "google/gemini-3.8-flash", "providers": ["google-ai-studio/flex", "google-ai-studio"],
                         "price": (0.75, 3.75)},
    # Teacher option for training data (Stage 6): a larger DeepSeek, MIT-licensed like V4.1 Flash.
    "deepseek-v4-pro": {"model": "deepseek/deepseek-v4-pro", "providers": ["deepinfra", "novita"], "price": (1.60, 3.20)},
}


def load_env() -> None:
    """Read KEY=VALUE lines from .env into the environment, without printing anything."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cost_usd(usage: dict, price: tuple[float, float] = (0.0, 0.0)) -> float:
    """The real cost when the API reported it, otherwise an estimate from the token counts."""
    if usage.get("cost") is not None:
        return float(usage["cost"])
    return (usage.get("input_tokens", 0) * price[0] + usage.get("output_tokens", 0) * price[1]) / 1e6


class OracleModel:
    name = "oracle"
    price = (0.0, 0.0)

    def step(self, case: dict, system_prompt: str, tools: list[dict]) -> dict:
        option = case["expect"]["options"][0]
        if option["kind"] == "tool_call":
            args = dict(option["args_exact"])
            for k in option["args_nonempty"]:
                args[k] = "x"
            for k, values in option["args_in"].items():
                args[k] = values[0]
            schema = next(t["function"]["parameters"] for t in tools if t["function"]["name"] == option["tool"])
            for k in schema.get("required", []):
                args.setdefault(k, "")
            return {"text": None, "tool_calls": [{"name": option["tool"], "arguments": args}], "raw": ""}
        parts = ["تمام"]
        parts += [t.split(":")[0] for t in option.get("must_mention_times", [])]
        parts += option["must_include_any"][:1]
        return {"text": " ".join(parts), "tool_calls": [], "raw": ""}


TEXT_TOOL_GUIDE = """You can use these tools. To call one, reply with only this line and nothing else:
<tool_call>{"name": "<tool name>", "arguments": {<arguments as JSON>}}</tool_call>
Tool results come back as <tool_result>...</tool_result>. Otherwise, reply to the customer normally.
Tools:
"""
TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def text_tool_messages(system_prompt: str, messages: list[dict], tools: list[dict]) -> list[dict]:
    """For models whose chat template has no tool support (e.g. ALLaM): tools are described
    in the system prompt, calls and results become plain text, and turns strictly alternate."""
    specs = "\n".join(json.dumps({"name": t["function"]["name"], "description": t["function"]["description"],
                                  "parameters": t["function"]["parameters"]}, ensure_ascii=False) for t in tools)
    out = [{"role": "system", "content": f"{system_prompt}\n\n{TEXT_TOOL_GUIDE}{specs}"}]
    for m in messages:
        if m.get("tool_calls"):
            role, text = "assistant", "\n".join(
                "<tool_call>" + json.dumps({"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                                           ensure_ascii=False) + "</tool_call>" for tc in m["tool_calls"])
        elif m["role"] == "tool":
            role, text = "user", f"<tool_result>{m['content']}</tool_result>"
        else:
            role, text = m["role"], m["content"]
        if out[-1]["role"] == role:  # merge, so user and assistant turns alternate
            out[-1]["content"] += "\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


def parse_text_tool_calls(text: str) -> tuple[str | None, list[dict]]:
    calls = []
    for body in TEXT_TOOL_CALL.findall(text):
        try:
            obj = json.loads(body)
            calls.append({"name": obj.get("name"), "arguments": obj.get("arguments", {})})
        except (json.JSONDecodeError, AttributeError):
            calls.append({"name": None, "arguments": body})  # malformed: the scorer marks it invalid
    rest = TEXT_TOOL_CALL.sub("", text).strip()
    return rest or None, calls


def uses_text_tools(tok) -> bool:
    """True when the model's chat template has no tool support, so tools have to be described in text."""
    probe_tool = [{"type": "function", "function": {"name": "t", "description": "d",
                                                     "parameters": {"type": "object", "properties": {}}}}]
    try:
        rendered = tok.apply_chat_template([{"role": "user", "content": "x"}], tools=probe_tool,
                                           tokenize=False, add_generation_prompt=True)
        return '"t"' not in rendered and "t{" not in rendered  # template silently dropped the tools
    except Exception:
        return True


class HFModel:
    """Local model in 4-bit NF4, greedy decoding, thinking switched off. Uses the model's own
    tool format when its chat template has one, otherwise describes the tools in text."""

    price = (0.0, 0.0)

    EMPTY_THOUGHT = "<|channel>thought\n<channel|>"  # what Gemma's template adds to a generation prompt with thinking off

    def __init__(self, repo_id: str, max_new_tokens: int = 256, adapter: str | None = None, no_thought: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = repo_id + (f"@{adapter}" if adapter else "") + ("~nothought" if no_thought else "")
        # Diagnostic only (~nothought): leave out the empty thinking block, so the prompt ends where the training
        # text did. The benchmark itself never uses this; it is how the train/test mismatch was checked (Oct 2026).
        self.no_thought = no_thought
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if adapter:  # the fine-tuned LoRA weights on top of the same 4-bit base (scripts/train.py)
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.text_tools = uses_text_tools(self.tok)

    def step(self, case: dict, system_prompt: str, tools: list[dict]) -> dict:
        if self.text_tools:
            return self.text_tool_step(case, system_prompt, tools)
        messages = [{"role": "system", "content": system_prompt}] + case["messages"]
        prompt = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                              enable_thinking=False, tokenize=False)
        if self.no_thought and prompt.endswith(self.EMPTY_THOUGHT):
            prompt = prompt[: -len(self.EMPTY_THOUGHT)]
        raw = self.tok.decode(self.generate_ids(prompt), skip_special_tokens=False)
        try:
            parsed = self.tok.parse_response(raw, prefix=prompt)
        except Exception as e:  # unparseable output counts as a plain-text reply
            parsed = {"content": raw, "parse_error": f"{type(e).__name__}: {e}"}
        calls = [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}}
                 for tc in parsed.get("tool_calls") or []]
        for c in calls:  # some parsers return arguments as a JSON string
            if isinstance(c["arguments"], str):
                try:
                    c["arguments"] = json.loads(c["arguments"])
                except json.JSONDecodeError:
                    pass
        return {"text": parsed.get("content"), "tool_calls": calls, "raw": raw,
                **({"parse_error": parsed["parse_error"]} if "parse_error" in parsed else {})}

    def text_tool_step(self, case: dict, system_prompt: str, tools: list[dict]) -> dict:
        messages = text_tool_messages(system_prompt, case["messages"], tools)
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        raw = self.tok.decode(self.generate_ids(prompt), skip_special_tokens=True)
        text, calls = parse_text_tool_calls(raw)
        return {"text": text, "tool_calls": calls, "raw": raw, "tool_format": "text"}

    def generate_ids(self, prompt: str) -> list[int]:
        """Greedy continuation of the prompt, as token ids (the end-of-turn token included)."""
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0, inputs["input_ids"].shape[1]:].tolist()


class LlamaCppModel(HFModel):
    """A GGUF build behind llama.cpp's llama-server. Everything but generation is HFModel's: the same tokenizer renders
    the same prompt, which is sent as token ids; the reply comes back as token ids and is parsed the same way."""

    def __init__(self, url: str, repo_id: str, max_new_tokens: int = 256):
        from transformers import AutoTokenizer

        self.url = url.rstrip("/")
        self.name = f"llamacpp:{self.url}@{repo_id}"
        self.max_new_tokens, self.no_thought = max_new_tokens, False
        self.tok = AutoTokenizer.from_pretrained(repo_id)
        self.text_tools = uses_text_tools(self.tok)
        eos = self.gen_eos(repo_id)
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])

    @staticmethod
    def gen_eos(repo_id: str):
        from transformers import GenerationConfig
        return GenerationConfig.from_pretrained(repo_id).eos_token_id

    def generate_ids(self, prompt: str) -> list[int]:
        import urllib.request
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        body = {"prompt": ids, "n_predict": self.max_new_tokens, "temperature": 0, "top_k": 1, "return_tokens": True,
                "cache_prompt": True}
        req = urllib.request.Request(f"{self.url}/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            res = json.loads(r.read())
        out = list(res.get("tokens") or [])
        # llama-server leaves out the token it stopped on; Hugging Face's generate keeps it. Put it back, so the two
        # parse the same text: for Gemma a tool call ends on <|tool_response>, a reply on <turn|>; for ALLaM </s>
        # (skipped when decoding anyway).
        if res.get("stop_type") == "eos" and (not out or out[-1] not in self.stop_ids):
            text = self.tok.decode(out, skip_special_tokens=False)
            if self.text_tools:
                out.append(self.tok.eos_token_id)
            elif text.rstrip().endswith("<tool_call|>"):
                out.append(self.tok.convert_tokens_to_ids("<|tool_response>"))
            else:
                out.append(self.tok.convert_tokens_to_ids("<turn|>"))
        return out


class LlamaChatModel:
    """llama-server's OpenAI-style chat API: what an app such as LM Studio gets, template and tool parser included."""

    price = (0.0, 0.0)

    def __init__(self, url: str, repo_id: str, max_new_tokens: int = 256):
        from transformers import AutoTokenizer
        self.url, self.max_new_tokens = url.rstrip("/"), max_new_tokens
        self.name = f"llamachat:{self.url}@{repo_id}"
        self.text_tools = uses_text_tools(AutoTokenizer.from_pretrained(repo_id))

    def chat(self, messages: list[dict], tools: list[dict] | None) -> dict:
        import urllib.request
        body = {"messages": messages, "temperature": 0, "top_k": 1, "max_tokens": self.max_new_tokens}
        if tools:
            body.update(tools=tools, chat_template_kwargs={"enable_thinking": False})
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["choices"][0]["message"]

    def step(self, case: dict, system_prompt: str, tools: list[dict]) -> dict:
        if self.text_tools:
            msg = self.chat(text_tool_messages(system_prompt, case["messages"], tools), None)
            text, calls = parse_text_tool_calls(msg.get("content") or "")
            return {"text": text, "tool_calls": calls, "raw": msg.get("content") or "", "tool_format": "text"}
        messages = [{"role": "system", "content": system_prompt}]
        for m in case["messages"]:
            if m.get("tool_calls"):  # OpenAI style: arguments as a JSON string
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"],
                     "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}} for tc in m["tool_calls"]]})
            else:
                messages.append({k: v for k, v in m.items() if k in ("role", "content", "tool_call_id", "name")})
        msg = self.chat(messages, tools)
        calls = []
        for tc in msg.get("tool_calls") or []:
            args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                pass
            calls.append({"name": tc["function"]["name"], "arguments": args})
        return {"text": (msg.get("content") or "").strip() or None, "tool_calls": calls, "raw": msg.get("content") or ""}


class OllamaModel:
    """A GGUF build served by Ollama, through its chat API: what an Ollama user gets, template and parser included."""

    price = (0.0, 0.0)

    def __init__(self, model: str, repo_id: str, url: str = "http://127.0.0.1:11434", max_new_tokens: int = 256):
        from transformers import AutoTokenizer
        self.model, self.url, self.max_new_tokens = model, url, max_new_tokens
        self.name = f"ollama:{model}@{repo_id}"
        self.text_tools = uses_text_tools(AutoTokenizer.from_pretrained(repo_id))

    def chat(self, messages: list[dict], tools: list[dict] | None) -> dict:
        import urllib.request
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": 0, "top_k": 1, "seed": 0, "num_predict": self.max_new_tokens, "num_ctx": 4096}}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body, ensure_ascii=False).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]

    def step(self, case: dict, system_prompt: str, tools: list[dict]) -> dict:
        if self.text_tools:
            msg = self.chat(text_tool_messages(system_prompt, case["messages"], tools), None)
            text, calls = parse_text_tool_calls(msg.get("content") or "")
            return {"text": text, "tool_calls": calls, "raw": msg.get("content") or "", "tool_format": "text"}
        messages = [{"role": "system", "content": system_prompt}]
        for m in case["messages"]:
            if m.get("tool_calls"):
                messages.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                    {"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for tc in m["tool_calls"]]})
            elif m["role"] == "tool":
                messages.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        msg = self.chat(messages, tools)
        calls = [{"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or {}}
                 for tc in msg.get("tool_calls") or []]
        return {"text": (msg.get("content") or "").strip() or None, "tool_calls": calls, "raw": msg.get("content") or ""}


class OpenRouterModel:
    """An API competitor through OpenRouter's OpenAI-compatible endpoint."""

    def __init__(self, competitor: str):
        import openai

        load_env()
        spec = COMPETITORS[competitor]
        self.openai = openai
        self.name, self.model, self.providers, self.price = competitor, spec["model"], spec["providers"], spec["price"]
        self.client = openai.OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"),
                                    base_url="https://openrouter.ai/api/v1")

    @staticmethod
    def convert(system_prompt: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system_prompt}]
        for m in messages:
            if m.get("tool_calls"):
                out.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                    for tc in m["tool_calls"]]})
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def step(self, case: dict, system_prompt: str, tools: list[dict]) -> dict:
        try:
            r = self.client.chat.completions.create(
                model=self.model, max_tokens=API_MAX_TOKENS, tools=tools,
                messages=self.convert(system_prompt, case["messages"]),
                extra_body={
                    "provider": {"order": self.providers, "only": self.providers, "data_collection": "deny"},
                    "reasoning": {"enabled": True},
                    "usage": {"include": True},
                })
        except self.openai.APIStatusError as e:
            return {"error": f"HTTP {e.status_code}"}
        except self.openai.APIConnectionError:
            return {"error": "connection error"}
        if not r.choices:
            return {"error": "empty response"}
        msg = r.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = tc.function.arguments  # kept as text; the scorer marks it invalid
            calls.append({"name": tc.function.name, "arguments": args})
        extra = (r.usage.model_extra or {}) if r.usage else {}
        return {"text": (msg.content or "").strip() or None, "tool_calls": calls, "raw": msg.content or "",
                "stop_reason": r.choices[0].finish_reason,
                "provider": (r.model_extra or {}).get("provider"),
                "usage": {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens,
                          "reasoning_tokens": getattr(r.usage.completion_tokens_details, "reasoning_tokens", None)
                          if r.usage.completion_tokens_details else None,
                          "cost": extra.get("cost")} if r.usage else {}}


def load_model(spec: str):
    if spec == "oracle":
        return OracleModel()
    if spec.startswith("hf:"):  # hf:<repo> or hf:<repo>@<LoRA adapter folder> (Stage 7); ~nothought: diagnostic
        body, no_thought = (spec[3:-len("~nothought")], True) if spec.endswith("~nothought") else (spec[3:], False)
        repo, _, adapter = body.partition("@")
        return HFModel(repo, adapter=adapter or None, no_thought=no_thought)
    if spec.startswith("llamacpp:"):  # llamacpp:<url>@<repo_id>
        url, _, repo = spec[len("llamacpp:"):].rpartition("@")
        return LlamaCppModel(url, repo)
    if spec.startswith("llamachat:"):  # llamachat:<url>@<repo_id>
        url, _, repo = spec[len("llamachat:"):].rpartition("@")
        return LlamaChatModel(url, repo)
    if spec.startswith("ollama:"):  # ollama:<model>@<repo_id>
        model, _, repo = spec[len("ollama:"):].rpartition("@")
        return OllamaModel(model, repo)
    if spec.startswith("or:"):
        return OpenRouterModel(spec[3:])
    raise ValueError(f"unknown model {spec!r} (use oracle, hf:<repo>, or or:<{'|'.join(COMPETITORS)}>)")
