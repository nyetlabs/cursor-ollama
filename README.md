# Cursor Ollama Proxy

An OpenAI-compatible proxy that lets [Cursor](https://cursor.com) talk to a local [Ollama](https://ollama.com) server. Cursor sends `/v1/chat/completions` requests; this process translates them to Ollama's native `/api/chat` API and translates the reply back, including tool calls.

## Credit

This repository started from [danilofalcao/cursor-deepseek](https://github.com/danilofalcao/cursor-deepseek), which proxies Cursor to DeepSeek, OpenRouter, and Ollama. The upstream Ollama path does not work properly with Cursor out of the box, so the proxy here is a full rewrite aimed only at Ollama. Thanks to Danilo Falcao for the original project and for the idea of putting a small OpenAI-compatible server in front of local models.

## What it translates

- `POST /v1/chat/completions` becomes `POST {OLLAMA_API_ENDPOINT}/chat`.
- `GET /v1/models` returns the models Ollama has. The list's `default` field, and `"default": true` on that model object, name `DEFAULT_MODEL`.
- Message content may be a string or an array of text parts. Other content part types are rejected.
- `temperature` and `max_tokens` are sent as Ollama `options.temperature` and `options.num_predict`.
- OpenAI `tools` and legacy `functions` are forwarded as Ollama tools. `tool_choice` is left for Ollama to decide.
- Native Ollama `tool_calls` are returned as OpenAI `tool_calls`, with `function.arguments` as a JSON string.
- JSON tool calls that a model writes into assistant text are pulled out and returned as OpenAI tool calls when they match a tool in the request. Native tool calls are kept when Ollama already returned them.
- Streaming clients receive an OpenAI server-sent event stream. The proxy reads the full Ollama reply first, then emits that stream.

The server listens on `LISTEN_ADDRESS`, which defaults to `127.0.0.1:9000`. It does not check `Authorization`. The Apache vhost does, using the key from Cursor's "Use OpenAI API Key" setting.

The model name on the chat request is the model Ollama runs, and the same name is written back on the completion. `DEFAULT_MODEL` fills in only when that name is empty. With `DEFAULT_MODEL` unset, the empty-name fallback is `llama2`, or the `-model` flag when one was passed.

Before calling Ollama, the proxy checks that name against Ollama's model list. An unknown name returns HTTP 404 and the names Ollama has. A name without a tag also matches the same name with `:latest`. The process exits at startup when `DEFAULT_MODEL` is not in that list, or when the list cannot be read. `GET /v1/models` repeats that check and returns HTTP 500 if the default model is no longer present.

## Requirements

- [Ollama](https://ollama.com) running and reachable from this process, with the model you want already pulled
- Go 1.19 or newer to run from source, or Docker to run the image

## Configuration

Copy `env.example` to `.env`. The proxy loads `.env` at startup, and `.env` is gitignored.

```bash
cp env.example .env
```

`env.example` sets:

```bash
LISTEN_ADDRESS=127.0.0.1:9000
OLLAMA_API_ENDPOINT=http://localhost:11434/api
DEFAULT_MODEL=gpt-oss:20b
```

Commented alternatives in that file are `deepseek-coder-v2:16b` and `qwen2.5-coder:14b`. Uncomment one of those, or set another model you have pulled.

`LISTEN_ADDRESS` defaults to `127.0.0.1:9000` when unset. `OLLAMA_API_ENDPOINT` defaults to `http://localhost:11434/api` when unset. `DEFAULT_MODEL` defaults to `llama2` when unset, and is used only when a request omits a model. A `-model` argument is used only when `DEFAULT_MODEL` is unset:

```bash
go run . -model gpt-oss:20b
```

The endpoint in `env.example` is for a proxy running on the host. From a container, point it at the host's Ollama server:

```bash
OLLAMA_API_ENDPOINT=http://host.docker.internal:11434/api
```

## Run

```bash
make
make run
```

The process logs the model, Ollama endpoint, and listen address, then serves that address. With the example configuration that is `http://127.0.0.1:9000`. That address is for local checks. Cursor does not call it. See [Cursor](#cursor).

`--debug` still calls Ollama, then appends each Cursor request and the response it would have returned to `cursor-ollama-debug.log`. A chat completion is answered with a finished message whose text is `ok`, so Cursor does not retry or run a tool. `/v1/models` is still the real model list.

```bash
go run . --debug
```

Confirm the proxy and Ollama on the host:

```bash
curl http://127.0.0.1:9000/v1/models
curl http://127.0.0.1:11434/api/tags
```

## Docker

The image builds `cursor-ollama.go` and sets `LISTEN_ADDRESS=:9000`. The `-e` flag keeps that address when `.env` sets the host loopback. The host mapping below still binds only on localhost:

```bash
docker build -t cursor-ollama .
docker run --rm -p 127.0.0.1:9000:9000 --env-file .env -e LISTEN_ADDRESS=:9000 cursor-ollama
```

On Docker Desktop, `host.docker.internal` reaches Ollama on the host. On Linux, add `--add-host=host.docker.internal:host-gateway` or set `OLLAMA_API_ENDPOINT` to an address the container can route.

## Apache

`apache2.conf.example` is an HTTPS reverse proxy in front of the process on `127.0.0.1:9000`. It still matches this server. Copy it into an Apache site, set `ServerName` and the certificate paths, and replace `EXAMPLE_TOKEN`.

Enable `ssl`, `proxy`, `proxy_http`, and `authz_core`. Apache allows a request only when the header is exactly `Authorization: Bearer EXAMPLE_TOKEN`. Requests straight to `127.0.0.1:9000` skip that check.

**The HTTPS server must be visible on the public internet for Cursor's API routing to work.** Cursor sends these requests from its own servers. A loopback address, a private LAN address, or any host that is not reachable from the internet will not receive them.

`ProxyPass http://localhost:9000/` follows the default `LISTEN_ADDRESS`. Change it if the proxy listens somewhere else. A container published with `-p 127.0.0.1:9000:9000` still uses this `ProxyPass`.

`ProxyTimeout 600` covers the wait for Ollama. The proxy writes its response only after that generation finishes, so Apache sees a quiet backend until then.

These directives keep that finished `text/event-stream` body from being compressed or held on a pooled backend connection:

- `proxy-nokeepalive`
- `proxy-sendchunked`
- `no-gzip`
- `proxy-initial-not-pooled`

The proxy sends `X-Accel-Buffering: no` for nginx. Apache ignores that header, so the directives above are what applies here. `ProxyPreserveHost` and `ProxyPassReverse` do not change this server's responses.

## Cursor

Open Cursor Settings, then Models, then API Keys.

- Turn on "Use OpenAI API Key" and set the OpenAI API key to the same value as `EXAMPLE_TOKEN` in the Apache vhost. Cursor sends it as `Authorization: Bearer EXAMPLE_TOKEN`.
- Turn on "Override OpenAI Base URL".
- Set "Base URL" to the public HTTPS endpoint, including `/v1`:

```text
https://ollama.example.org/v1
```

That host must be visible on the public internet. Cursor's API routing does not call an address that only this machine can reach.

Add each Ollama model id you want, such as `gpt-oss:20b`, to Cursor's model list and select it for Chat or Agent. The proxy forwards that id to Ollama. The id has to be a model already pulled on the Ollama server.

## Tests

```bash
go test ./...
```

## License

GNU General Public License v2.0. See [LICENSE.md](LICENSE.md).

The upstream project, [cursor-deepseek](https://github.com/danilofalcao/cursor-deepseek), is also licensed under GPLv2.
