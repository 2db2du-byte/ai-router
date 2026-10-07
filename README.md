# joshua

Chat with local Ollama models from the terminal; every message is automatically sent to the best installed model.

```
joshua                      # interactive session (/help inside for commands)
joshua "question"           # one-shot
cat notes.txt | joshua "summarize this"
joshua --model mistral:7b   # skip routing
```

A small fast model (`qwen3:1.7b`) classifies each message into a category, and the
category decides the model: chat, knowledge, code, reasoning, writing, summarize,
agent (runs commands / edits files on this machine, asking before anything that isn't
read-only), and vision (when the message mentions an image path).

## Internet

Questions that need current info (news, prices, "latest version", weather...) are routed to a `web`
category whose model can search and read pages; agent mode has the same tools. Weather comes from
wttr.in. Search tries the laptop's SearXNG first (http://127.0.0.1:8888, started on demand with `searxng start`),
then DuckDuckGo, then Bing, then Wikipedia, dropping results that don't match the
query; free engines block bots now and then, so for reliable search add a free ollama.com API key:

```
joshua --search-key YOUR_KEY     # saved to ~/.config/ai/config.json (readable only by you)
joshua --search-key              # remove it
```

Every model is told today's date.

## Updating

`joshua --update` (or `/update` inside) re-pulls every installed model; only changed parts download.
Ollama itself is a system package and updates with the rest of the system (`omarchy update`).

## Memory

- **Conversation:** `joshua` picks up your last conversation when you reopen it. `/new` starts fresh and
  archives the old one in `~/.local/state/ai/conversations/`. One-shot `ai "..."` runs don't touch it.
- **Long-term facts:** after each answer, `qwen3:8b` checks in the background for lasting facts about
  you (name, preferences, setup, projects...) and saves them to `~/.local/share/ai/memory.md`; every
  model is given them. You can also say "remember that ...", and anything starting with "from now on",
  "always" or "never" is saved as a standing instruction.
- `/memory` lists what's saved, `/forget N` or `/forget TEXT` removes entries; the file is plain
  text and safe to edit by hand.

Change the model for a category in `~/.config/ai/config.json`:

```json
{ "routes": { "reasoning": { "model": "deepseek-r1:8b", "think": true } } }
```

If the Ollama service isn't running, `joshua` starts it with `sudo systemctl start ollama`.

## License

MIT. Built by Rabbid Raccoon with Claude. Use it, change it, share it.
