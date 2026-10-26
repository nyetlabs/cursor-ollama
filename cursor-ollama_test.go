package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func catalogFor(names ...string) toolCatalog {
	tools := make([]Tool, 0, len(names))
	for _, name := range names {
		tools = append(tools, Tool{Type: "function", Function: Function{Name: name}})
	}
	return catalogFromTools(tools)
}

func objectParams(properties []string, required []string) map[string]any {
	props := map[string]any{}
	for _, name := range properties {
		props[name] = map[string]any{"type": "string"}
	}
	requiredValues := make([]any, 0, len(required))
	for _, name := range required {
		requiredValues = append(requiredValues, name)
	}
	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   requiredValues,
	}
}

func TestModelInList(t *testing.T) {
	names := []string{"gpt-oss:20b", "qwen2.5-coder:14b", "llama3:latest"}
	if !modelInList("gpt-oss:20b", names) || !modelInList("llama3", names) {
		t.Fatal("expected listed models and an untagged :latest alias to match")
	}
	if modelInList("missing", names) || modelInList("llama3:13b", names) {
		t.Fatal("expected unknown and differently tagged models to miss")
	}
	msg := unknownModelMessage("missing", names)
	if !strings.Contains(msg, "missing") || !strings.Contains(msg, "gpt-oss:20b") {
		t.Fatalf("unexpected error: %s", msg)
	}
	if empty := unknownModelMessage("missing", nil); !strings.Contains(empty, "no models") {
		t.Fatalf("unexpected empty-list error: %s", empty)
	}
	listed, ok := listedModelName("llama3", names)
	if !ok || listed != "llama3:latest" {
		t.Fatalf("listed name = %q ok=%v", listed, ok)
	}
	payload, err := modelsList(names, "llama3")
	if err != nil {
		t.Fatal(err)
	}
	if payload["default"] != "llama3:latest" {
		t.Fatalf("default marker = %#v", payload["default"])
	}
	if _, err := modelsList(names, "missing"); err == nil {
		t.Fatal("expected an error when the default model is absent")
	}
}

func TestExtractJSONToolCallFormats(t *testing.T) {
	names := catalogFor("Read", "Shell", "Say")
	cases := []struct {
		name    string
		content string
		calls   int
		first   string
		path    string
		rest    string
	}{
		{
			name:    "object",
			content: `{"name":"Read","arguments":{"path":"a.go"}}`,
			calls:   1,
			first:   "Read",
			path:    "a.go",
		},
		{
			name:    "fenced",
			content: "```json\n{\"name\":\"Read\",\"arguments\":{\"path\":\"a.go\"}}\n```",
			calls:   1,
			first:   "Read",
			path:    "a.go",
		},
		{
			name:    "hermes",
			content: "<tool_call>\n{\"name\":\"Read\",\"parameters\":{\"path\":\"a.go\"}}\n</tool_call>",
			calls:   1,
			first:   "Read",
			path:    "a.go",
		},
		{
			name:    "llama marker",
			content: `[TOOL_CALLS][{"name":"Read","arguments":{"path":"a.go"}}]`,
			calls:   1,
			first:   "Read",
			path:    "a.go",
		},
		{
			name:    "nested function",
			content: `{"function":{"name":"functions.Read","arguments":"{\"path\":\"a.go\"}"}}`,
			calls:   1,
			first:   "Read",
			path:    "a.go",
		},
		{
			name:    "prose around call",
			content: "I'll read it.\n{\"tool\":\"Read\",\"tool_input\":{\"path\":\"a.go\"}}\n",
			calls:   1,
			first:   "Read",
			path:    "a.go",
			rest:    "I'll read it.",
		},
		{
			name:    "flattened args",
			content: `{"action":"Shell","command":"ls"}`,
			calls:   1,
			first:   "Shell",
		},
		{
			name:    "qwen empty arguments",
			content: `{ "name": "Say", "arguments": }`,
			calls:   1,
			first:   "Say",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls, rest := extractToolCallsFromContent(tc.content, names)
			if len(calls) != tc.calls {
				t.Fatalf("got %d calls, want %d (%q)", len(calls), tc.calls, rest)
			}
			if calls[0].Function.Name != tc.first {
				t.Fatalf("name %q, want %q", calls[0].Function.Name, tc.first)
			}
			if tc.path != "" {
				var args map[string]any
				if err := json.Unmarshal(calls[0].Function.Arguments, &args); err != nil {
					t.Fatal(err)
				}
				if args["path"] != tc.path {
					t.Fatalf("path %#v, want %q", args["path"], tc.path)
				}
			}
			if rest != tc.rest {
				t.Fatalf("remainder %q, want %q", rest, tc.rest)
			}
		})
	}
}

func TestExtractLeavesUnrelatedJSON(t *testing.T) {
	content := `{"name":"NotATool","arguments":{"path":"a.go"}}`
	calls, rest := extractToolCallsFromContent(content, catalogFor("Read"))
	if len(calls) != 0 {
		t.Fatalf("unexpected calls: %+v", calls)
	}
	if rest != content {
		t.Fatalf("content changed to %q", rest)
	}

	schema := `{"name":"Read","description":"read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}`
	calls, rest = extractToolCallsFromContent(schema, catalogFor("Read"))
	if len(calls) != 0 || rest != schema {
		t.Fatalf("schema echo was treated as a call: calls=%d rest=%q", len(calls), rest)
	}

	calls, rest = extractToolCallsFromContent(`{"name":"Read","arguments":{"path":"a.go"}}`, toolCatalog{})
	if len(calls) != 0 || !strings.Contains(rest, "Read") {
		t.Fatalf("translated a call without offered tools")
	}
}

func TestResolvePrefersNativeToolCalls(t *testing.T) {
	native := []OllamaToolCall{makeOllamaCall("Read", json.RawMessage(`{"path":"a.go"}`))}
	text, calls := resolveAssistantOutput(`{"name":"Read","arguments":{"path":"a.go"}}`, native, catalogFor("Read"))
	if text != "" || len(calls) != 1 || calls[0].Function.Name != "Read" {
		t.Fatalf("text=%q calls=%+v", text, calls)
	}
}

func TestOpenAIToolCallArgumentsAreStrings(t *testing.T) {
	calls := toOpenAIToolCalls([]OllamaToolCall{
		makeOllamaCall("Read", json.RawMessage(`{"path":"a.go"}`)),
	}, 10)
	encoded, err := json.Marshal(calls)
	if err != nil {
		t.Fatal(err)
	}
	var parsed []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(encoded, &parsed); err != nil {
		t.Fatalf("arguments were not a JSON string: %v body=%s", err, encoded)
	}
	if parsed[0].Type != "function" || parsed[0].Function.Name != "Read" || parsed[0].ID != "call_10_0" {
		t.Fatalf("unexpected tool call: %+v", parsed[0])
	}
	if parsed[0].Function.Arguments != `{"path":"a.go"}` {
		t.Fatalf("arguments %s", parsed[0].Function.Arguments)
	}
}

func TestInferBareArgumentsAsSmallestMatchingTool(t *testing.T) {
	cat := catalogFromTools([]Tool{
		{Type: "function", Function: Function{Name: "Grep", Parameters: objectParams([]string{"pattern", "path", "glob"}, []string{"pattern"})}},
		{Type: "function", Function: Function{Name: "Glob", Parameters: objectParams([]string{"pattern", "target_directory"}, []string{"pattern"})}},
	})
	calls, rest := extractToolCallsFromContent(`{"pattern":"*"}`, cat)
	if rest != "" || len(calls) != 1 || calls[0].Function.Name != "Glob" {
		t.Fatalf("rest=%q calls=%+v", rest, calls)
	}
	if argumentsString(calls[0].Function.Arguments) != `{"pattern":"*"}` {
		t.Fatalf("arguments %s", calls[0].Function.Arguments)
	}

	calls, rest = extractToolCallsFromContent(`{"pattern":"*","path":"src"}`, cat)
	if len(calls) != 1 || calls[0].Function.Name != "Grep" {
		t.Fatalf("expected Grep, rest=%q calls=%+v", rest, calls)
	}

	tied := catalogFromTools([]Tool{
		{Type: "function", Function: Function{Name: "Glob", Parameters: objectParams([]string{"pattern"}, []string{"pattern"})}},
		{Type: "function", Function: Function{Name: "Grep", Parameters: objectParams([]string{"pattern"}, []string{"pattern"})}},
	})
	calls, rest = extractToolCallsFromContent(`{"pattern":"*"}`, tied)
	if len(calls) != 0 || rest != `{"pattern":"*"}` {
		t.Fatalf("ambiguous arguments were converted: rest=%q calls=%+v", rest, calls)
	}
}
