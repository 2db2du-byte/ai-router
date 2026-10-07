#!/usr/bin/env python3
"""joshua - talk to your local Ollama models; each message is routed to the best one.

Usage:
  joshua                  interactive session
  joshua "question"       one-shot answer
  cat file | joshua "prompt"  include piped text in the prompt
  joshua --model NAME ... skip routing and use NAME
  joshua --update         update all installed models
  joshua --search-key KEY use an ollama.com API key for reliable web search
"""

import atexit
import html
import html.parser
import json
import os
import re
import readline  # noqa: F401  (gives input() line editing and history)
import shutil
import signal
import stat
import subprocess
import threading
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
DEFAULT_HOST = HOST in ("http://127.0.0.1:11434", "http://localhost:11434")

CONFIG_PATH = os.path.expanduser("~/.config/ai/config.json")
HISTORY_PATH = os.path.expanduser("~/.local/state/ai/history")
OLLAMA_LOG = os.path.expanduser("~/.local/state/ai/ollama.log")
OLLAMA_PID = os.path.expanduser("~/.local/state/ai/ollama.pid")  # set while joshua owns the server

# category -> which model handles it, plus the description the router sees.
ROUTES = {
    "chat":      {"model": "llama3.2:3b",      "think": False,
                  "desc": "casual conversation, greetings, small talk, quick simple questions"},
    "knowledge": {"model": "qwen3:8b",         "think": False,
                  "desc": "explaining concepts, facts, how things work, advice, recommendations"},
    "code":      {"model": "qwen2.5-coder:7b", "think": False,
                  "desc": "writing, fixing, reviewing or explaining code, scripts, regex, SQL, shell one-liners"},
    "reasoning": {"model": "qwen3:8b",         "think": True,
                  "desc": "math, logic puzzles, step-by-step problem solving, planning, comparing tradeoffs"},
    "writing":   {"model": "llama3.1:8b",      "think": False,
                  "desc": "emails, essays, stories, poems, rewriting, editing, translating, changing tone"},
    "summarize": {"model": "qwen3.5:4b",       "think": False,
                  "desc": "summarizing or extracting information from a long pasted text"},
    "agent":     {"model": "qwen3:8b",         "think": False,
                  "desc": "actually DOING something on this computer: run commands, create/edit/find files, "
                          "install or set up software, check system state, automate a task"},
    "web":       {"model": "qwen3:8b",         "think": False,
                  "desc": "needs the INTERNET: news, weather, prices, sports scores, recent events, latest versions, "
                          "anything after 2024, or when the user says to search/look it up online"},
    "vision":    {"model": "gemma3:4b",        "think": False,
                  "desc": "questions about an attached image"},
}
ROUTER_MODEL = "qwen3:1.7b"
SEARCH_KEY = os.environ.get("OLLAMA_API_KEY", "")  # optional: ollama.com key for reliable web search
MEMORY_MODEL = "qwen3:8b"  # best recall with little noise in testing; usually already loaded
NUM_CTX = 8192
KEEP_ALIVE = "15m"
MAX_AGENT_STEPS = 15

ROUTER_EXAMPLES = """\
"hi there!" -> chat
"what year did WW2 end" -> knowledge
"what's the difference between RAM and storage" -> knowledge
"how does wifi work" -> knowledge
"should I get a laptop or a desktop for school, compare them" -> reasoning
"fix this python error: TypeError: 'NoneType' object is not subscriptable" -> code
"why does my bash script say command not found" -> code
"a bat and ball cost $1.10, the bat costs $1 more than the ball, what does the ball cost" -> reasoning
"should I rent or buy a house? weigh the pros and cons" -> reasoning
"rewrite this to sound more professional: ..." -> writing
"what's in my Documents folder" -> agent
"how much RAM am I using right now" -> agent
"install ffmpeg" -> agent
"create a folder called notes in my home dir with a todo.txt" -> agent
"set up a node project in ~/Projects/site" -> agent
"which process is using the most CPU" -> agent
"what's my name?" -> chat
"what do you remember about me" -> chat
"from now on answer in plain English" -> chat
"what's the weather in Denver today" -> web
"who won the game last night" -> web
"what's the latest version of Firefox" -> web
"look up reviews for the Framework laptop" -> web"""

IMAGE_RE = re.compile(r"(?:^|\s)(\S+\.(?:png|jpe?g|webp|gif|bmp))\b", re.I)

DIM, BOLD, CYAN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"
if not sys.stdout.isatty():
    DIM = BOLD = CYAN = YELLOW = RED = RESET = ""


def load_config():
    global ROUTER_MODEL, SEARCH_KEY
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return
    except json.JSONDecodeError as e:
        print(f"{RED}ignoring {CONFIG_PATH}: {e}{RESET}", file=sys.stderr)
        return
    ROUTER_MODEL = cfg.get("router_model", ROUTER_MODEL)
    SEARCH_KEY = cfg.get("ollama_api_key") or SEARCH_KEY
    for cat, override in cfg.get("routes", {}).items():
        ROUTES.setdefault(cat, {"model": "", "think": False, "desc": cat}).update(override)


# ---------------------------------------------------------------- ollama api

def api(path, payload=None, stream=False, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    if not stream:
        return json.load(resp)
    return (json.loads(line) for line in resp if line.strip())


def server_up():
    try:
        api("/api/version", timeout=1)
        return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_server():
    if server_up():
        return
    if not DEFAULT_HOST:
        sys.exit(f"{RED}Ollama isn't reachable at {HOST}{RESET}")
    # Run Ollama as the user (models in ~/.ollama/models), so starting it never needs a password.
    # stop_server() shuts it down again when the last joshua quits; it never starts at boot.
    print(f"{DIM}Starting Ollama...{RESET}", file=sys.stderr)
    os.makedirs(os.path.dirname(OLLAMA_LOG), exist_ok=True)
    with open(OLLAMA_LOG, "ab") as log:
        proc = subprocess.Popen(["ollama", "serve"], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                start_new_session=True)
    with open(OLLAMA_PID, "w") as f:
        f.write(str(proc.pid))
    for _ in range(60):
        if server_up():
            return
        time.sleep(0.5)
    sys.exit(f"{RED}Ollama started but isn't answering (log: {OLLAMA_LOG}){RESET}")


def other_joshuas():
    """Other running joshua sessions (they still need the server)."""
    me = os.getpid()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if int(pid) != me and len(argv) > 1 and b"python" in argv[0] and argv[1].endswith((b"/joshua", b"/ai.py")):
            return True
    return False


def stop_server():
    """Stop Ollama if joshua started it and this is the last joshua running."""
    try:
        with open(OLLAMA_PID) as f:
            pid = int(f.read())
    except (OSError, ValueError):
        return  # started some other way (e.g. by hand): leave it alone
    if other_joshuas():
        return
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            if b"ollama" in f.read():
                os.kill(pid, signal.SIGTERM)
        for _ in range(50):  # let it unload models and exit cleanly (usually 1-2 s)
            try:
                if os.waitpid(pid, os.WNOHANG)[0]:  # we started it this run: reap it
                    break
            except ChildProcessError:
                os.kill(pid, 0)  # started by an earlier joshua: raises once it's gone
            time.sleep(0.1)
    except OSError:
        pass  # gone
    os.remove(OLLAMA_PID)


def installed_models():
    return {m["name"] for m in api("/api/tags")["models"]}


def update_models():
    """Re-pull every installed model; Ollama only downloads the parts that changed."""
    before = {m["name"]: m["digest"] for m in api("/api/tags")["models"]}
    failed = []
    for i, name in enumerate(sorted(before), 1):
        label = f"[{i}/{len(before)}] {name}"
        try:
            for chunk in api("/api/pull", {"model": name, "stream": True}, stream=True, timeout=3600):
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                status = chunk.get("status", "")
                if chunk.get("total") and chunk.get("completed") is not None:
                    status = f"downloading {100 * chunk['completed'] // chunk['total']}%"
                print(f"\r\033[K{DIM}{label}: {status}{RESET}", end="", flush=True)
        except Exception as e:
            failed.append(name)
            print(f"\r\033[K{RED}{label}: failed - {e}{RESET}")
    print("\r\033[K", end="")
    after = {m["name"]: m["digest"] for m in api("/api/tags")["models"]}
    updated = [n for n in before if n in after and after[n] != before[n]]
    for name in updated:
        print(f"⬆️  updated {name}")
    ok = len(before) - len(updated) - len(failed)
    print(f"{DIM}{ok} model{'s' if ok != 1 else ''} already up to date"
          + (f", {len(failed)} failed (check your internet)" if failed else "") + f"{RESET}")


# ---------------------------------------------------------------- routing

def route(text, history, last_cat, has_image):
    if has_image:
        return "vision", "image attached"
    cats = [c for c in ROUTES if c != "vision"]
    menu = "\n".join(f"- {c}: {ROUTES[c]['desc']}" for c in cats)
    context = ""
    if last_cat and history:
        prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        context = (f"\nThe previous message was routed to '{last_cat}' and said: {prev[:300]!r}. "
                   f"If the new message is a follow-up to it (e.g. 'make it shorter', 'why?', 'try again'), "
                   f"keep '{last_cat}' unless the new message clearly needs something else.")
    prompt = (f"Classify the user's message into exactly one category.\n{menu}\n\n"
              f"Examples:\n{ROUTER_EXAMPLES}\n\n"
              f"Rule: if answering requires looking at or changing THIS computer (its files, folders, disk, "
              f"installed programs, settings, running processes), the category is agent - even if it sounds "
              f"like a question. General questions about how things work in the world are NOT agent.{context}\n\n"
              f"User message:\n{text[:2000]}")
    try:
        r = api("/api/chat", {
            "model": ROUTER_MODEL, "stream": False, "think": False, "keep_alive": "30m",
            "messages": [{"role": "user", "content": prompt}],
            "format": {"type": "object", "properties": {"category": {"type": "string", "enum": cats}},
                       "required": ["category"]},
            "options": {"temperature": 0, "num_predict": 20},
        }, timeout=120)
        cat = json.loads(r["message"]["content"])["category"]
        if cat in ROUTES:
            return cat, "router"
    except Exception as e:  # router trouble shouldn't block answering
        return "knowledge", f"router failed: {e}"
    return "knowledge", "router fallback"


# ---------------------------------------------------------------- agent tools

TOOLS = [
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a bash command on the user's Linux computer (Arch Linux, Omarchy/Hyprland) and return its output. "
                       "The user must approve each command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file with the given content. The user must approve.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List the entries of a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]

WEB_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the internet (DuckDuckGo). Returns titles, links and short snippets. Use for anything "
                       "current or that you're unsure about.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather and 3-day forecast for a place (city, zip code, airport code). "
                       "Use this instead of web_search for weather.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Download a web page and return its readable text. Use it to read a search result in full.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]

WEB_SYSTEM = (
    "You answer questions using the internet. Search first, open the most promising result or two with fetch_url "
    "when the snippets aren't enough, then answer plainly and mention the source sites. If the results don't "
    "answer the question, say so instead of guessing."
)

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"


class _TextExtractor(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"}

    def __init__(self):
        super().__init__()
        self.parts, self.skipping, self.title, self.in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        if tag == "title":
            self.in_title = True
        if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skipping:
            self.parts.append(data)

    def text(self):
        lines = (" ".join(line.split()) for line in "".join(self.parts).splitlines())
        return "\n".join(line for line in lines if line)


def http_get(url, timeout=15, limit=2_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read(limit)
        charset = resp.headers.get_content_charset() or "utf-8"
        return body.decode(charset, errors="replace"), resp.headers.get_content_type()


def _strip_tags(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", s)).split())


def _search_duckduckgo(query):
    page, _ = http_get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}))
    results = []
    for block in page.split('class="result__body"')[1:]:
        link = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not link:
            continue
        href = html.unescape(link.group(1))
        if "uddg=" in href:  # DuckDuckGo wraps links in a redirect
            href = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["uddg"][0]
        if "duckduckgo.com/y.js" in href:  # ads
            continue
        snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        results.append((_strip_tags(link.group(2)), href, _strip_tags(snippet.group(1)) if snippet else ""))
    return results


def _search_bing(query):
    page, _ = http_get("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query}))
    results = []
    for block in page.split('class="b_algo"')[1:]:
        block = re.sub(r"<link[^>]*>|<style.*?</style>|<script.*?</script>", "", block, flags=re.S)
        link = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not link:
            continue
        href = html.unescape(link.group(1))
        target = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("u", [""])[0]
        if target.startswith("a1"):  # Bing hides the real link as base64 in a redirect
            import base64
            href = base64.urlsafe_b64decode(target[2:] + "=" * (-len(target[2:]) % 4)).decode(errors="replace")
        snippet = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        results.append((_strip_tags(link.group(2)), href, _strip_tags(snippet.group(1)) if snippet else ""))
    return results


SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")  # the laptop's private search engine


def _search_searxng(query):
    """The laptop's SearXNG (started on demand with `searxng start`). Private and reliable; no key."""
    url = SEARXNG_URL + "/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.load(resp)
            break
        except (urllib.error.URLError, OSError):
            if attempt == 2 or not shutil.which("searxng"):
                return []  # couldn't start it - fall through to the next engine
            print(f"{DIM}starting SearXNG...{RESET}", file=sys.stderr)
            subprocess.run(["searxng", "start"], capture_output=True, timeout=90)
    return [(r.get("title", ""), r.get("url", ""), " ".join((r.get("content") or "").split())[:500])
            for r in data.get("results", [])]


def _search_ollama(query):
    if not SEARCH_KEY:
        return []
    req = urllib.request.Request("https://ollama.com/api/web_search",
                                 data=json.dumps({"query": query, "max_results": 8}).encode(),
                                 headers={"Authorization": f"Bearer {SEARCH_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.load(resp)
    return [(r.get("title", ""), r.get("url", ""), " ".join(r.get("content", "").split())[:500])
            for r in data.get("results", [])]


def _search_wikipedia(query):
    data = json.loads(http_get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "format": "json", "srsearch": " ".join(_key_words(query)) or query}))[0])
    return [(r["title"], "https://en.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
             _strip_tags(r["snippet"])) for r in data["query"]["search"]]


SEARCH_STOPWORDS = {"what", "when", "where", "which", "who", "whom", "how", "why", "the", "and", "for", "with",
                    "about", "latest", "current", "today", "best", "does", "did", "that", "this", "from", "version",
                    "release", "date", "news", "are", "was", "were", "is", "can", "you", "tell", "find", "search",
                    "look", "online", "please", "give", "show"}


def _key_words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in SEARCH_STOPWORDS]


def _relevant(results, query):
    """Drop results that miss the query's key words - engines feed bots junk sometimes."""
    stems = {w[:5] for w in _key_words(query)}  # crude stemming: starred/starring/stars all match
    stems = {w for w in stems if not w.isdigit()} or stems  # years like "2026" match everything
    if not stems:
        return results
    need = len(stems) if len(stems) <= 2 else (len(stems) + 1) // 2
    return [r for r in results if len(stems & {w[:5] for w in _key_words(" ".join(r))}) >= need]


def web_search(query, max_results=6):
    # The laptop's SearXNG first; free search engines block or poison bot traffic, so fall through.
    for engine in (_search_searxng, _search_ollama, _search_duckduckgo, _search_bing, _search_wikipedia):
        try:
            results = engine(query)
            if engine not in (_search_searxng, _search_ollama):  # these return real results, not bot bait
                results = _relevant(results, query)
            results = results[:max_results]
        except Exception:
            continue
        if len(results) >= 2 or (results and engine in (_search_searxng, _search_ollama, _search_wikipedia)):
            return "\n\n".join(f"{title}\n{url}\n{snippet}" for title, url, snippet in results)
    return "Search failed: every search engine refused or returned nothing. Tell the user to try again later."


def get_weather(location):
    data = json.loads(http_get("https://wttr.in/" + urllib.parse.quote(location) + "?format=j1")[0])
    now, area = data["current_condition"][0], data["nearest_area"][0]
    place = ", ".join(x for x in (area["areaName"][0]["value"], area["region"][0]["value"],
                                  area["country"][0]["value"]) if x)
    lines = [f"Weather for {place} (source: wttr.in)",
             f"Now: {now['weatherDesc'][0]['value']}, {now['temp_F']}°F / {now['temp_C']}°C "
             f"(feels like {now['FeelsLikeF']}°F), humidity {now['humidity']}%, wind {now['windspeedMiles']} mph"]
    for day in data["weather"]:
        rain = max(int(h.get("chanceofrain", 0)) for h in day["hourly"])
        lines.append(f"{day['date']}: high {day['maxtempF']}°F / low {day['mintempF']}°F, "
                     f"{day['hourly'][4]['weatherDesc'][0]['value']}, up to {rain}% chance of rain")
    return "\n".join(lines)


def fetch_url(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    body, ctype = http_get(url)
    if "html" not in ctype:
        return clip(body, 6000)
    parser = _TextExtractor()
    parser.feed(body)
    return clip(f"Title: {parser.title.strip()}\n\n{parser.text()}", 6000)


AGENT_SYSTEM = (
    "You are an assistant that gets things done on the user's Linux computer (Arch Linux with Omarchy/Hyprland, "
    f"bash shell, home directory {os.path.expanduser('~')}). Use the tools to inspect the system and carry out the "
    "task step by step. Prefer small, safe commands; look before you change things. Never use sudo unless the task "
    "truly needs it. You can also search the web and read pages when you need instructions or current info. "
    "When done, reply with a short summary of what you did and the result."
)


# Commands that only look at things. A command is auto-approved only if every piece of
# a pipeline/chain starts with one of these and nothing redirects, substitutes or deletes.
SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "fd", "df", "du", "free", "nproc", "lscpu",
    "lsblk", "lspci", "lsusb", "uname", "whoami", "hostname", "date", "uptime", "ps", "pgrep", "which",
    "type", "echo", "pwd", "stat", "file", "sort", "uniq", "cut", "tr", "tree", "sensors", "id", "groups",
    "locale", "printenv", "realpath", "basename", "dirname", "column", "journalctl", "awk",
}
SAFE_SUBCOMMANDS = {
    "pacman": ("-Q", "-Qi", "-Qs", "-Ss", "-Si", "-Ql", "-Qe", "-Qm"),
    "systemctl": ("status", "is-active", "is-enabled", "list-units", "list-unit-files", "list-timers"),
    "hyprctl": ("monitors", "clients", "workspaces", "activewindow", "devices", "version", "configerrors",
                "getoption", "layers", "binds"),
    "ip": ("a", "addr", "address", "route", "r", "link"),
    "git": ("status", "log", "diff", "branch", "show", "remote"),
    "ollama": ("list", "ls", "ps", "show"),
}
UNSAFE_PATTERNS = re.compile(r">|<\(|`|\$\(|-delete\b|-exec|-ok\b|-fprint|-fls\b|--output|\bsort\b.*\s-o|\bsudo\b|\bsystem\s*\(")


def is_read_only(cmd):
    if UNSAFE_PATTERNS.search(cmd):
        return False
    for part in re.split(r"\|\||&&|[|;]", cmd):
        words = part.split()
        if not words:
            continue
        if words[0] in SAFE_COMMANDS:
            continue
        if words[0] == "systemctl" and words[1:2] == ["--user"]:
            words = words[:1] + words[2:]
        subs = SAFE_SUBCOMMANDS.get(words[0])
        if subs and len(words) > 1 and words[1] in subs:
            continue
        return False
    return True


def confirm(question):
    try:
        with open("/dev/tty") as tty:
            print(f"{YELLOW}{question} [y/N] {RESET}", end="", flush=True)
            return tty.readline().strip().lower() in ("y", "yes")
    except OSError:
        print(f"{RED}(no terminal to ask for approval - skipped){RESET}")
        return False


DECLINED = ("The user said NO to this action. Do not retry it or try a workaround. Stop using tools and "
            "ask the user how they would like to proceed.")


def clip(s, n=8000):
    return s if len(s) <= n else s[:n] + f"\n... ({len(s) - n} more characters cut)"


def run_tool(name, args):
    try:
        if name == "run_command":
            cmd = args["command"]
            if is_read_only(cmd):
                print(f"{CYAN}$ {cmd}{RESET} {DIM}(read-only, auto-approved){RESET}")
            else:
                print(f"{CYAN}$ {cmd}{RESET}")
                if not confirm("Run this command?"):
                    return DECLINED
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300,
                               executable="/bin/bash", cwd=os.path.expanduser("~"))
            out = (p.stdout + p.stderr).strip()
            print(DIM + clip(out, 2000) + RESET)
            return clip(f"exit code {p.returncode}\n{out}")
        if name == "read_file":
            print(f"{DIM}reading {args['path']}{RESET}")
            with open(os.path.expanduser(args["path"])) as f:
                return clip(f.read())
        if name == "write_file":
            path = os.path.expanduser(args["path"])
            content = args["content"]
            print(f"{CYAN}write {path} ({len(content)} chars):{RESET}\n{DIM}{clip(content, 1500)}{RESET}")
            if os.path.exists(path):
                print(f"{YELLOW}(this file already exists and will be overwritten){RESET}")
            if not confirm("Write this file?"):
                return DECLINED
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"wrote {path}"
        if name == "web_search":
            print(f"{DIM}🔎 searching: {args['query']}{RESET}")
            return web_search(args["query"])
        if name == "get_weather":
            print(f"{DIM}🌤️  checking the weather for {args['location']}{RESET}")
            return get_weather(args["location"])
        if name == "fetch_url":
            print(f"{DIM}🌐 reading {args['url']}{RESET}")
            return fetch_url(args["url"])
        if name == "list_dir":
            path = os.path.expanduser(args.get("path") or ".")
            print(f"{DIM}listing {path}{RESET}")
            entries = sorted(os.listdir(path))
            lines = [e + "/  (directory)" if os.path.isdir(os.path.join(path, e)) else e for e in entries]
            return clip(f"Directory {path} contains {len(entries)} entries:\n" + "\n".join(lines))
        return f"unknown tool {name}"
    except subprocess.TimeoutExpired:
        return "command timed out after 300s"
    except Exception as e:
        return f"error: {e}"


# ---------------------------------------------------------------- generation

def stream_chat(model, messages, think=False, tools=None):
    """Stream a reply to the terminal; return (text, tool_calls)."""
    payload = {"model": model, "messages": messages, "stream": True, "keep_alive": KEEP_ALIVE,
               "options": {"num_ctx": NUM_CTX}}
    if tools:
        payload["tools"] = tools
    if think:
        payload["think"] = True
    elif model.startswith(("qwen3", "deepseek-r1")):
        payload["think"] = False
    text, calls, thinking = "", [], False
    try:
        for chunk in api("/api/chat", payload, stream=True):
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            msg = chunk.get("message", {})
            if msg.get("thinking"):
                if not thinking:
                    print(f"{DIM}thinking... ", end="")
                    thinking = True
                print(msg["thinking"], end="", flush=True)
            if msg.get("content"):
                if thinking:
                    print(f"{RESET}\n")
                    thinking = False
                text += msg["content"]
                print(msg["content"], end="", flush=True)
            calls += msg.get("tool_calls") or []
    finally:
        if thinking:
            print(RESET)
    if text:
        print()
    return text, calls


def answer(model, cat, history, text, images):
    user_msg = {"role": "user", "content": text}
    if images:
        user_msg["images"] = images
    route_cfg = ROUTES.get(cat, {})
    memory = memory_prompt()
    # Models only know the date they were trained; tell them the real one.
    system = f"Today is {time.strftime('%A, %B %-d, %Y')}." + ("\n\n" + memory if memory else "")
    tools = {"agent": TOOLS + WEB_TOOLS, "web": WEB_TOOLS}.get(cat)
    if not tools:
        msgs = [{"role": "system", "content": system}] + history + [user_msg]
        reply, _ = stream_chat(model, msgs, think=route_cfg.get("think", False))
        return user_msg, reply

    # Tool loop: the model calls tools until it gives a plain answer.
    role = AGENT_SYSTEM if cat == "agent" else WEB_SYSTEM
    msgs = [{"role": "system", "content": role + "\n\n" + system}] + history + [user_msg]
    for _ in range(MAX_AGENT_STEPS):
        reply, calls = stream_chat(model, msgs, tools=tools)
        if not calls:
            return user_msg, reply
        msgs.append({"role": "assistant", "content": reply, "tool_calls": calls})
        for call in calls:
            fn = call["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = run_tool(fn["name"], args).strip() or "(no output)"
            # Small models ignore bare results; say plainly where this came from.
            shown_args = ", ".join(f"{k}={v!r}" for k, v in args.items() if k != "content")
            msgs.append({"role": "tool", "tool_name": fn["name"], "tool_call_id": call.get("id"),
                         "content": f"Result of {fn['name']}({shown_args}):\n{result}"})
    print(f"{YELLOW}(stopped after {MAX_AGENT_STEPS} steps){RESET}")
    return user_msg, reply


def encode_images(text):
    import base64
    images = []
    for path in IMAGE_RE.findall(text):
        p = os.path.expanduser(path)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())
    return images


# ---------------------------------------------------------------- memory

MEMORY_PATH = os.path.expanduser("~/.local/share/ai/memory.md")
CONVERSATION_PATH = os.path.expanduser("~/.local/state/ai/conversation.json")
ARCHIVE_DIR = os.path.expanduser("~/.local/state/ai/conversations")
MAX_FACTS = 80
REMEMBER_RE = re.compile(r"^\s*(?:please\s+)?(?:remember|keep in mind|note)(?:\s+that)?[:,]?\s+(.+)$", re.I | re.S)
STANDING_RE = re.compile(r"^\s*(?:please\s+)?(?:from now on|going forward|always|never)\b", re.I)
_memory_lock = threading.Lock()


def load_facts():
    try:
        with open(MEMORY_PATH) as f:
            return [line[2:].strip() for line in f if line.startswith("- ") and line[2:].strip()]
    except FileNotFoundError:
        return []


def save_facts(facts):
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        f.write("# What `joshua` remembers about you. Edit freely: one '- ' line per fact.\n\n")
        f.writelines(f"- {fact}\n" for fact in facts)


FILLER_WORDS = {"the", "user", "user's", "users", "said", "is", "are", "a", "an", "and", "of", "to", "for", "in",
                "on", "with", "their", "they", "has", "have", "my", "i", "i'm", "am", "that", "this", "as", "at"}


def key_words(fact):
    words = (w.strip(".'-") for w in re.findall(r"[a-z0-9'+#.-]+", fact.lower().replace("’", "'")))
    return {w for w in words if w and w not in FILLER_WORDS}


def is_covered(fact, facts):
    """True if the fact's key words are (almost) all in one existing fact - a rephrased duplicate."""
    words = key_words(fact)
    return bool(words) and any(len(words & key_words(f)) / len(words) >= 0.75 for f in facts)


def add_facts(new):
    with _memory_lock:
        facts = load_facts()
        added = []
        for fact in new:
            fact = " ".join(fact.split())
            if fact and not is_covered(fact, facts + added):
                added.append(fact)
        if added:
            save_facts((facts + added)[-MAX_FACTS:])
        return added


def memory_prompt():
    facts = load_facts()
    if not facts:
        return None
    return ("Long-term memory - things you know about the user from earlier conversations. Use them when "
            "relevant (e.g. address them by name, follow their preferences); don't list them back unprompted.\n"
            + "\n".join(f"- {f}" for f in facts))


def extract_facts(user_text, reply):
    """Ask the small model whether this exchange revealed anything worth keeping; save it."""
    known = "\n".join(f"- {f}" for f in load_facts()) or "(nothing yet)"
    prompt = (
        "You keep the long-term memory of a personal assistant. Read the exchange and list NEW lasting facts "
        "about the USER: their name, work, preferences (including how they like answers), their computer "
        "setup, projects they're working on, people or pets in their life, plans and goals.\n"
        "Write each fact as a short sentence starting with 'The user'.\n"
        "Do NOT save: general knowledge, what the assistant said, what the user is asking about, looking "
        "for or interested in right now (\"The user wants to know...\", \"The user is looking for...\"), "
        "one-off tasks, temporary state (disk space, what's running), or facts already known. Save the "
        "lasting fact behind a question instead (\"how do I stop my dog Max barking\" -> the user has a dog "
        "named Max). Only save what the user clearly STATED - never guess or infer. A format request for one "
        "reply (\"in one sentence\", \"briefly\") is not a lasting preference unless they say always or in "
        "general. Most messages contain nothing worth "
        "saving - return an empty list for those.\n\n"
        "Examples:\n"
        "User: what's the capital of France -> []\n"
        "User: how much RAM am I using -> []\n"
        "User: write me a poem about the sea -> []\n"
        "User: I'm Alex, a nurse, trying to learn Spanish -> [\"The user's name is Alex.\", "
        "\"The user works as a nurse.\", \"The user is learning Spanish.\"]\n"
        "User: can you stop using so many bullet points -> [\"The user prefers answers without many bullet points.\"]\n"
        "User: my cat Milo is sick, what should I do -> [\"The user has a cat named Milo.\"]\n"
        "User: I run Fedora on a ThinkPad -> [\"The user runs Fedora on a ThinkPad.\"]\n\n"
        f"Already known:\n{known}\n\nUser said:\n{user_text[:1500]}\n\nAssistant replied:\n{reply[:600]}"
    )
    try:
        r = api("/api/chat", {
            "model": MEMORY_MODEL, "stream": False, "think": False, "keep_alive": KEEP_ALIVE,
            "messages": [{"role": "user", "content": prompt}],
            "format": {"type": "object", "required": ["facts"], "properties": {
                "facts": {"type": "array", "maxItems": 3, "items": {"type": "string"}}}},
            "options": {"temperature": 0, "num_predict": 200},
        }, timeout=120)
        facts = json.loads(r["message"]["content"]).get("facts", [])
        return add_facts([f for f in facts if isinstance(f, str) and 8 < len(f) < 200])
    except Exception:
        return []


def save_conversation(history, last_cat):
    os.makedirs(os.path.dirname(CONVERSATION_PATH), exist_ok=True)
    tmp = CONVERSATION_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"updated": time.time(), "last_cat": last_cat, "history": history}, f)
    os.replace(tmp, CONVERSATION_PATH)


def load_conversation():
    try:
        with open(CONVERSATION_PATH) as f:
            data = json.load(f)
        return data.get("history", []), data.get("last_cat"), data.get("updated")
    except (FileNotFoundError, json.JSONDecodeError):
        return [], None, None


def archive_conversation():
    if not os.path.exists(CONVERSATION_PATH):
        return
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.replace(CONVERSATION_PATH, os.path.join(ARCHIVE_DIR, time.strftime("%Y-%m-%d_%H%M%S") + ".json"))


def ago(ts):
    s = time.time() - ts
    for unit, n in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if s >= n:
            k = int(s // n)
            return f"{k} {unit}{'s' if k > 1 else ''} ago"
    return "just now"


# ---------------------------------------------------------------- session

HELP = f"""{BOLD}Just type what you want.{RESET} Each message goes to the best model automatically.
  /model NAME   always use NAME (e.g. /model mistral:7b)     /model auto   back to automatic
  /use CAT      force a category for your next messages ({', '.join(ROUTES)})
  /routes       show which model handles what        /update   update all models to their latest versions
  /new          start a fresh conversation (the old one is archived)
  /memory       show what it remembers about you     /forget N|TEXT   remove a memory
  "remember that ..."  save something to long-term memory
  /exit (or Ctrl-D)    quit - your conversation is kept for next time
  \"\"\"           start/end a multi-line message (for pasting)
  Mention an image path (e.g. ~/Pictures/x.png) to ask about a picture.
  In agent mode every command and file write asks for your OK first."""


class Session:
    def __init__(self, pinned_model=None, persist=True):
        self.persist = persist  # one-shot runs use memory but don't touch the saved conversation
        self.history, self.last_cat, self.updated = load_conversation() if persist else ([], None, None)
        self.pinned_model = pinned_model
        self.forced_cat = None
        self.installed = installed_models()
        self.memory_thread = None
        self.new_facts = []

    def remember_in_background(self, text, reply):
        def work():
            self.new_facts += extract_facts(text, reply)
        self.memory_thread = threading.Thread(target=work, daemon=True)
        self.memory_thread.start()

    def show_new_facts(self, wait=False):
        if wait and self.memory_thread:
            self.memory_thread.join()
        for fact in self.new_facts:
            print(f"{DIM}💾 remembered: {fact}{RESET}")
        self.new_facts = []

    def record(self, user_msg, reply, cat):
        self.history += [user_msg, {"role": "assistant", "content": reply}]
        self.history = self.history[-40:]
        self.last_cat = cat
        if self.persist:
            save_conversation(self.history, cat)

    def ask(self, text):
        m = REMEMBER_RE.match(text)
        if m and len(text) < 400 and not text.rstrip().endswith("?"):
            added = add_facts([f"The user said: {m.group(1).strip()}"])
            reply = "Got it, I'll remember that." if added else "I already had that in my memory."
            print(f"{DIM}💾 {reply}{RESET}")
            self.record({"role": "user", "content": text}, reply, self.last_cat)
            return
        if STANDING_RE.match(text) and len(text) < 400:  # a standing instruction: keep it, then answer
            for fact in add_facts([f"The user said: {text.strip()}"]):
                print(f"{DIM}💾 remembered: {fact}{RESET}")
        images = encode_images(text)
        if self.pinned_model:
            cat, why, model = self.forced_cat or "manual", "pinned", self.pinned_model
        else:
            if self.forced_cat:
                cat, why = self.forced_cat, "forced"
            else:
                if self.last_cat is None:
                    print(f"{DIM}routing...{RESET}", end="\r", flush=True)
                cat, why = route(text, self.history, self.last_cat, bool(images))
            model = ROUTES[cat]["model"]
        if model not in self.installed:
            print(f"{RED}{model} isn't installed (ollama pull {model}); using {ROUTER_MODEL}{RESET}")
            model = ROUTER_MODEL
        print(f"\r\033[K{DIM}→ {cat} · {model}{' (' + why + ')' if why != 'router' else ''}{RESET}")
        try:
            user_msg, reply = answer(model, cat, self.history, text, images)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}(interrupted){RESET}")
            return
        except Exception as e:
            print(f"{RED}error from {model}: {e}{RESET}")
            return
        user_msg.pop("images", None)  # don't resend big images every turn
        self.record(user_msg, reply, cat)
        if reply.strip():
            self.remember_in_background(text, reply)

    def command(self, line):
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd in ("/exit", "/quit", "/q"):
            raise EOFError
        elif cmd == "/help":
            print(HELP)
        elif cmd in ("/new", "/clear"):
            archive_conversation()
            self.history, self.last_cat = [], None
            print(f"{DIM}fresh conversation (long-term memory is kept; the old chat is in {ARCHIVE_DIR}){RESET}")
        elif cmd == "/memory":
            facts = load_facts()
            if not facts:
                print(f"{DIM}nothing remembered yet - say \"remember that ...\"{RESET}")
            for i, fact in enumerate(facts, 1):
                print(f"  {DIM}{i:>2}.{RESET} {fact}")
            print(f"{DIM}stored in {MEMORY_PATH} (you can edit it){RESET}")
        elif cmd == "/remember":
            print(f"{DIM}💾 {'remembered' if add_facts([arg]) else 'already known'}{RESET}" if arg
                  else f"{RED}usage: /remember TEXT{RESET}")
        elif cmd == "/forget":
            with _memory_lock:
                facts = load_facts()
                if arg.isdigit() and 1 <= int(arg) <= len(facts):
                    gone = [facts.pop(int(arg) - 1)]
                elif arg:
                    gone = [f for f in facts if arg.lower() in f.lower()]
                    facts = [f for f in facts if f not in gone]
                else:
                    gone = []
                save_facts(facts)
            for fact in gone:
                print(f"{DIM}forgot: {fact}{RESET}")
            if not gone:
                print(f"{RED}usage: /forget NUMBER or /forget TEXT (see /memory){RESET}")
        elif cmd == "/update":
            update_models()
            self.installed = installed_models()
        elif cmd == "/routes":
            for c, r in ROUTES.items():
                print(f"  {BOLD}{c:<10}{RESET} {r['model']:<18} {DIM}{r['desc']}{RESET}")
            print(f"  {BOLD}{'router':<10}{RESET} {ROUTER_MODEL}")
        elif cmd == "/model":
            if not arg or arg == "auto":
                self.pinned_model = None
                print(f"{DIM}automatic model choice{RESET}")
            elif arg in self.installed or arg + ":latest" in self.installed:
                self.pinned_model = arg
                print(f"{DIM}using {arg} for everything (/model auto to undo){RESET}")
            else:
                print(f"{RED}not installed. Installed: {', '.join(sorted(self.installed))}{RESET}")
        elif cmd == "/use":
            if not arg or arg == "auto":
                self.forced_cat = None
                print(f"{DIM}automatic routing{RESET}")
            elif arg in ROUTES:
                self.forced_cat = arg
                print(f"{DIM}sending everything to '{arg}' (/use auto to undo){RESET}")
            else:
                print(f"{RED}categories: {', '.join(ROUTES)}{RESET}")
        else:
            print(f"{RED}unknown command; /help for help{RESET}")


def read_message():
    line = input(f"{BOLD}{CYAN}› {RESET}")
    if line.strip() != '"""':
        return line
    lines = []
    while True:
        more = input(f"{DIM}… {RESET}")
        if more.strip() == '"""':
            return "\n".join(lines)
        lines.append(more)


def save_search_key(key):
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}
    if key:
        cfg["ollama_api_key"] = key.strip()
    else:
        cfg.pop("ollama_api_key", None)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    fd = os.open(CONFIG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # only you can read it
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2)
    os.chmod(CONFIG_PATH, 0o600)
    print(f"search key {'saved' if key else 'removed'} ({CONFIG_PATH})")


def main():
    args = sys.argv[1:]
    if args[:1] in (["-h"], ["--help"]):
        print(__doc__.strip() + "\n\n" + HELP)
        return
    pinned = None
    if args[:1] == ["--model"] and len(args) > 1:
        pinned, args = args[1], args[2:]

    if args[:1] == ["--search-key"]:
        save_search_key(args[1] if len(args) > 1 else "")
        return

    load_config()
    ensure_server()
    if args[:1] == ["--update"]:
        update_models()
        return

    prompt = " ".join(args)
    mode = os.fstat(sys.stdin.fileno()).st_mode
    if stat.S_ISFIFO(mode) or stat.S_ISREG(mode):  # piped or redirected, e.g. cat file | joshua
        piped = sys.stdin.read()
        prompt = f"{prompt}\n\n{piped}" if prompt else piped
    if prompt.strip():
        session = Session(pinned, persist=False)
        session.ask(prompt)
        session.show_new_facts(wait=True)
        return

    session = Session(pinned)

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    try:
        readline.read_history_file(HISTORY_PATH)
    except OSError:
        pass
    print(f"{DIM}Local AI · type anything · /help for options · Ctrl-D to quit{RESET}")
    if session.history:
        turns = sum(m["role"] == "user" for m in session.history)
        print(f"{DIM}continuing your conversation from {ago(session.updated)} ({turns} messages) · /new for a fresh one{RESET}")
    if facts := load_facts():
        print(f"{DIM}remembering {len(facts)} thing{'s' if len(facts) > 1 else ''} about you · /memory to see{RESET}")
    while True:
        session.show_new_facts()
        try:
            line = read_message().strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue
        if not line:
            continue
        if line.startswith("/"):
            try:
                session.command(line)
            except EOFError:
                break
            continue
        session.ask(line)
        try:
            readline.write_history_file(HISTORY_PATH)
        except OSError:
            pass
    if session.memory_thread and session.memory_thread.is_alive():
        print(f"{DIM}saving memories...{RESET}")
    session.show_new_facts(wait=True)
    try:
        readline.write_history_file(HISTORY_PATH)
    except OSError:
        pass


if __name__ == "__main__":
    atexit.register(stop_server)
    # Closing the terminal (SIGHUP) or a kill (SIGTERM) should still stop Ollama; the conversation
    # is already saved after every answer.
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(0))
    try:
        main()
    except KeyboardInterrupt:
        print()
