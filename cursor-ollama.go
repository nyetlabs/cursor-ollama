package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"golang.org/x/net/http2"
)

const (
	ollamaEndpoint    = "http://localhost:11434/api"
	defaultModel      = "llama2"
	defaultListenAddr = "127.0.0.1:9000"
)

// Configuration structure
type Config struct {
	endpoint   string
	model      string
	listenAddr string
	debug      bool
}

var activeConfig Config

func init() {
	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: .env file not found or error loading it: %v", err)
	}
	activeConfig.endpoint = os.Getenv("OLLAMA_API_ENDPOINT")
	if activeConfig.endpoint == "" {
		activeConfig.endpoint = ollamaEndpoint
	}
	activeConfig.model = os.Getenv("DEFAULT_MODEL")
	if activeConfig.model == "" {
		activeConfig.model = defaultModel
		for i, arg := range os.Args {
			if arg == "-model" && i+1 < len(os.Args) {
				activeConfig.model = os.Args[i+1]
			}
		}
	}
	activeConfig.listenAddr = os.Getenv("LISTEN_ADDRESS")
	if activeConfig.listenAddr == "" {
		activeConfig.listenAddr = defaultListenAddr
	}
	activeConfig.debug = debugFlagSet()
	if activeConfig.debug {
		if err := openDebugLog(); err != nil {
			log.Fatalf("debug log: %v", err)
		}
	}
	log.Printf("Initialized with default model: %s, using endpoint %s, listening on %s", activeConfig.model, activeConfig.endpoint, activeConfig.listenAddr)
	if activeConfig.debug {
		log.Printf("Debug logging requests and would-be responses to %s; clients receive ok", debugLogName)
	}
}

// OpenAI-compatible request/response structures.
type ChatRequest struct {
	Model       string          `json:"model"`
	Messages    []Message       `json:"messages"`
	Stream      bool            `json:"stream"`
	Functions   []Function      `json:"functions,omitempty"`
	Tools       []Tool          `json:"tools,omitempty"`
	ToolChoice  json.RawMessage `json:"tool_choice,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	MaxTokens   *int            `json:"max_tokens,omitempty"`
}

type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

type Function struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type ToolCall struct {
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

// Ollama's /api/chat message format differs from OpenAI's format.
type OllamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content,omitempty"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
	ToolName  string           `json:"tool_name,omitempty"`
}

type OllamaToolCall struct {
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type OllamaRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Tools    []Tool          `json:"tools,omitempty"`
	Options  map[string]any  `json:"options,omitempty"`
}

type OllamaResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role      string           `json:"role"`
		Content   string           `json:"content"`
		ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason,omitempty"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	EvalCount       int    `json:"eval_count,omitempty"`
}

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)
	if err := requireDefaultModel(); err != nil {
		log.Fatalf("%s", err)
	}
	server := &http.Server{Addr: activeConfig.listenAddr, Handler: http.HandlerFunc(proxyHandler)}
	http2.ConfigureServer(server, &http2.Server{})
	log.Printf("Starting Ollama proxy server on %s", server.Addr)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func enableCors(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length")
	w.Header().Set("Access-Control-Allow-Credentials", "true")
}

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	log.Printf("Received request: %s %s", r.Method, r.URL.Path)
	if serveDebug(w, r) {
		return
	}
	serveProxy(w, r)
}

func serveProxy(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		enableCors(w)
		return
	}
	enableCors(w)
	switch r.URL.Path {
	case "/v1/chat/completions":
		handleChatCompletions(w, r)
	case "/v1/models":
		handleModelsRequest(w)
	default:
		http.Error(w, "Not found", http.StatusNotFound)
	}
}

func contentToText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return "", fmt.Errorf("unsupported message content: %s", raw)
	}
	var out strings.Builder
	for _, part := range parts {
		if part.Type != "text" && part.Type != "" {
			return "", fmt.Errorf("unsupported structured content part type %q; this proxy currently supports text parts only", part.Type)
		}
		out.WriteString(part.Text)
	}
	return out.String(), nil
}

func toOllamaMessages(messages []Message) ([]OllamaMessage, error) {
	out := make([]OllamaMessage, 0, len(messages))
	for _, msg := range messages {
		content, err := contentToText(msg.Content)
		if err != nil {
			return nil, fmt.Errorf("message role %q: %w", msg.Role, err)
		}
		om := OllamaMessage{Role: msg.Role, Content: content}
		if msg.Role == "tool" {
			om.ToolName = msg.Name
		}
		for _, call := range msg.ToolCalls {
			om.ToolCalls = append(om.ToolCalls, makeOllamaCall(call.Function.Name, call.Function.Arguments))
		}
		out = append(out, om)
	}
	return out, nil
}

func toOpenAIToolCalls(calls []OllamaToolCall, created int64) []map[string]any {
	out := make([]map[string]any, 0, len(calls))
	for i, call := range calls {
		if call.Function.Name == "" {
			continue
		}
		out = append(out, map[string]any{
			"index": i,
			"id":    fmt.Sprintf("call_%d_%d", created, i),
			"type":  "function",
			"function": map[string]any{
				"name":      call.Function.Name,
				"arguments": argumentsString(call.Function.Arguments),
			},
		})
	}
	return out
}

// argumentsString returns the JSON object string Cursor expects in function.arguments.
func argumentsString(raw json.RawMessage) string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return "{}"
	}
	if raw[0] == '"' {
		var encoded string
		if json.Unmarshal(raw, &encoded) == nil {
			return argumentsString(json.RawMessage(encoded))
		}
	}
	if json.Valid(raw) && raw[0] == '{' {
		var compact bytes.Buffer
		if json.Compact(&compact, raw) == nil {
			return compact.String()
		}
	}
	return "{}"
}

type toolSchema struct {
	name       string
	properties map[string]struct{}
	required   map[string]struct{}
}

type toolCatalog []toolSchema

func catalogFromTools(tools []Tool) toolCatalog {
	var cat toolCatalog
	for _, tool := range tools {
		if tool.Function.Name == "" {
			continue
		}
		props, required := map[string]struct{}{}, map[string]struct{}{}
		if schema := asMap(tool.Function.Parameters); schema != nil {
			for key := range asMap(schema["properties"]) {
				props[key] = struct{}{}
			}
			for _, item := range asSlice(schema["required"]) {
				if key, ok := item.(string); ok {
					required[key] = struct{}{}
				}
			}
		}
		cat = append(cat, toolSchema{name: tool.Function.Name, properties: props, required: required})
	}
	return cat
}

func asMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	encoded, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var m map[string]any
	if json.Unmarshal(encoded, &m) != nil {
		return nil
	}
	return m
}

func asSlice(v any) []any {
	switch t := v.(type) {
	case []any:
		return t
	case []string:
		out := make([]any, len(t))
		for i, s := range t {
			out[i] = s
		}
		return out
	default:
		return nil
	}
}

func matchToolName(name string, cat toolCatalog) (string, bool) {
	name = strings.TrimSpace(strings.TrimPrefix(name, "functions."))
	if name == "" {
		return "", false
	}
	var folded string
	for _, tool := range cat {
		if tool.name == name {
			return tool.name, true
		}
		if strings.EqualFold(tool.name, name) {
			if folded != "" && folded != tool.name {
				return "", false
			}
			folded = tool.name
		}
	}
	return folded, folded != ""
}

func makeOllamaCall(name string, args json.RawMessage) OllamaToolCall {
	var call OllamaToolCall
	call.Type = "function"
	call.Function.Name = name
	call.Function.Arguments = json.RawMessage(argumentsString(args))
	return call
}

func isSchema(v any) bool {
	obj := asMap(v)
	return obj != nil && obj["type"] != nil && obj["properties"] != nil
}

func skipArgKey(key string) bool {
	switch key {
	case "name", "function", "tool", "tool_name", "action", "type", "id", "index", "description", "tool_calls",
		"arguments", "parameters", "args", "input", "tool_input", "action_input":
		return true
	default:
		return false
	}
}

func callFromName(name string, obj map[string]any, cat toolCatalog) (OllamaToolCall, bool) {
	matched, ok := matchToolName(name, cat)
	if !ok {
		return OllamaToolCall{}, false
	}
	var args any
	found := false
	for _, key := range []string{"arguments", "parameters", "args", "input", "tool_input", "action_input"} {
		if obj[key] != nil {
			args, found = obj[key], true
			break
		}
	}
	if !found {
		args = map[string]any{}
		for key, value := range obj {
			if !skipArgKey(key) {
				args.(map[string]any)[key] = value
			}
		}
	}
	if isSchema(args) {
		return OllamaToolCall{}, false
	}
	encoded, _ := json.Marshal(args)
	return makeOllamaCall(matched, encoded), true
}

func inferBareArguments(obj map[string]any, cat toolCatalog) (OllamaToolCall, bool) {
	if len(obj) == 0 || isSchema(obj) {
		return OllamaToolCall{}, false
	}
	for _, key := range []string{"name", "tool", "tool_name", "action", "function", "tool_calls"} {
		if _, ok := obj[key]; ok {
			return OllamaToolCall{}, false
		}
	}
	keys := make([]string, 0, len(obj))
	for key := range obj {
		keys = append(keys, key)
	}
	best, bestSize, tied := "", -1, false
	for _, tool := range cat {
		if len(tool.properties) == 0 || !argsFit(keys, tool) {
			continue
		}
		if bestSize == -1 || len(tool.properties) < bestSize {
			best, bestSize, tied = tool.name, len(tool.properties), false
		} else if len(tool.properties) == bestSize {
			tied = true
		}
	}
	if best == "" || tied {
		return OllamaToolCall{}, false
	}
	encoded, _ := json.Marshal(obj)
	return makeOllamaCall(best, encoded), true
}

func argsFit(keys []string, tool toolSchema) bool {
	present := map[string]struct{}{}
	for _, key := range keys {
		if _, ok := tool.properties[key]; !ok {
			return false
		}
		present[key] = struct{}{}
	}
	for key := range tool.required {
		if _, ok := present[key]; !ok {
			return false
		}
	}
	return true
}

func parseToolCallValue(v any, cat toolCatalog) []OllamaToolCall {
	switch typed := v.(type) {
	case []any:
		var calls []OllamaToolCall
		for _, item := range typed {
			calls = append(calls, parseToolCallValue(item, cat)...)
		}
		return calls
	case map[string]any:
		if calls := parseToolCallValue(typed["tool_calls"], cat); len(calls) > 0 {
			return calls
		}
		if fn, ok := typed["function"].(map[string]any); ok {
			if calls := parseToolCallValue(fn, cat); len(calls) > 0 {
				return calls
			}
		}
		for _, key := range []string{"name", "tool", "tool_name", "action", "function"} {
			name, ok := typed[key].(string)
			if !ok || strings.TrimSpace(name) == "" {
				continue
			}
			if call, ok := callFromName(name, typed, cat); ok {
				return []OllamaToolCall{call}
			}
			return nil
		}
		if call, ok := inferBareArguments(typed, cat); ok {
			return []OllamaToolCall{call}
		}
	}
	return nil
}

func appendUniqueCalls(dst, src []OllamaToolCall) []OllamaToolCall {
	for _, call := range src {
		if strings.TrimSpace(call.Function.Name) == "" {
			continue
		}
		call = makeOllamaCall(strings.TrimSpace(call.Function.Name), call.Function.Arguments)
		dup := false
		for _, existing := range dst {
			if existing.Function.Name == call.Function.Name && argumentsString(existing.Function.Arguments) == argumentsString(call.Function.Arguments) {
				dup = true
				break
			}
		}
		if !dup {
			dst = append(dst, call)
		}
	}
	return dst
}

func cleanupToolMarkup(content string) string {
	for _, marker := range []string{"```json", "```JSON", "```", "<tool_call>", "</tool_call>", "[TOOL_CALLS]"} {
		content = strings.ReplaceAll(content, marker, "")
	}
	return strings.TrimSpace(content)
}

// repairEmptyToolArguments fills in the empty value qwen2.5-coder leaves after "arguments":.
func repairEmptyToolArguments(content string) string {
	keys := []string{`"arguments"`, `"parameters"`, `"tool_input"`, `"action_input"`, `"args"`, `"input"`}
	var out strings.Builder
	for i := 0; i < len(content); {
		matched := ""
		for _, key := range keys {
			if strings.HasPrefix(content[i:], key) {
				matched = key
				break
			}
		}
		if matched == "" {
			out.WriteByte(content[i])
			i++
			continue
		}
		j := skipSpace(content, i+len(matched))
		if j >= len(content) || content[j] != ':' {
			out.WriteString(content[i:j])
			i = j
			continue
		}
		j = skipSpace(content, j+1)
		if j < len(content) && (content[j] == ',' || content[j] == '}' || content[j] == ']') {
			out.WriteString(matched)
			out.WriteString(":{}")
			i = j
			continue
		}
		out.WriteString(content[i:j])
		i = j
	}
	return out.String()
}

func skipSpace(content string, i int) int {
	for i < len(content) {
		switch content[i] {
		case ' ', '\n', '\t', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// extractToolCallsFromContent turns JSON tool calls embedded in model text into calls Cursor can run.
func extractToolCallsFromContent(content string, cat toolCatalog) ([]OllamaToolCall, string) {
	if len(cat) == 0 || strings.TrimSpace(content) == "" {
		return nil, content
	}
	original := content
	content = repairEmptyToolArguments(content)
	var calls []OllamaToolCall
	var spans [][2]int
	for i := 0; i < len(content); i++ {
		if content[i] != '{' && content[i] != '[' {
			continue
		}
		dec := json.NewDecoder(strings.NewReader(content[i:]))
		var v any
		if dec.Decode(&v) != nil {
			continue
		}
		parsed := parseToolCallValue(v, cat)
		if len(parsed) == 0 {
			continue
		}
		end := i + int(dec.InputOffset())
		calls = appendUniqueCalls(calls, parsed)
		spans = append(spans, [2]int{i, end})
		i = end - 1
	}
	if len(calls) == 0 {
		return nil, original
	}
	var remainder strings.Builder
	prev := 0
	for _, span := range spans {
		if span[0] > prev {
			remainder.WriteString(content[prev:span[0]])
		}
		prev = span[1]
	}
	remainder.WriteString(content[prev:])
	return calls, cleanupToolMarkup(remainder.String())
}

func resolveAssistantOutput(content string, native []OllamaToolCall, cat toolCatalog) (string, []OllamaToolCall) {
	native = appendUniqueCalls(nil, native)
	calls, rest := extractToolCallsFromContent(content, cat)
	if len(native) == 0 {
		return rest, calls
	}
	if strings.TrimSpace(rest) == strings.TrimSpace(content) {
		return strings.TrimSpace(content), native
	}
	return rest, native
}

func readOllamaChat(r io.Reader) (string, []OllamaToolCall, error) {
	var content strings.Builder
	var calls []OllamaToolCall
	dec := json.NewDecoder(r)
	for {
		var msg OllamaResponse
		if err := dec.Decode(&msg); err != nil {
			if err == io.EOF {
				return content.String(), calls, nil
			}
			return content.String(), calls, err
		}
		content.WriteString(msg.Message.Content)
		calls = appendUniqueCalls(calls, msg.Message.ToolCalls)
	}
}

var ollamaTagsClient = &http.Client{Timeout: 10 * time.Second}

func ollamaModelNames() ([]string, error) {
	resp, err := ollamaTagsClient.Get(fmt.Sprintf("%s/tags", activeConfig.endpoint))
	if err != nil {
		return nil, fmt.Errorf("could not read Ollama's model list: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("could not read Ollama's model list: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("Ollama's model list returned %s: %s", resp.Status, bytes.TrimSpace(body))
	}
	var tags struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.Unmarshal(body, &tags); err != nil {
		return nil, fmt.Errorf("could not parse Ollama's model list: %w", err)
	}
	var names []string
	seen := map[string]struct{}{}
	add := func(name string) {
		name = strings.TrimSpace(name)
		if name == "" {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}
	for _, model := range tags.Models {
		add(model.Name)
		add(model.Model)
	}
	sort.Strings(names)
	return names, nil
}

func listedModelName(requested string, names []string) (string, bool) {
	requested = strings.TrimSpace(requested)
	for _, name := range names {
		if name == requested {
			return name, true
		}
	}
	if !strings.Contains(requested, ":") {
		alias := requested + ":latest"
		for _, name := range names {
			if name == alias {
				return name, true
			}
		}
	}
	return "", false
}

func modelInList(requested string, names []string) bool {
	_, ok := listedModelName(requested, names)
	return ok
}

func requireDefaultModel() error {
	names, err := ollamaModelNames()
	if err != nil {
		return err
	}
	listed, ok := listedModelName(activeConfig.model, names)
	if !ok {
		return fmt.Errorf("%s", unknownModelMessage(activeConfig.model, names))
	}
	activeConfig.model = listed
	return nil
}

func modelsList(names []string, defaultModel string) (map[string]any, error) {
	listed, ok := listedModelName(defaultModel, names)
	if !ok {
		return nil, fmt.Errorf("%s", unknownModelMessage(defaultModel, names))
	}
	created := time.Now().Unix()
	data := make([]any, 0, len(names))
	for _, name := range names {
		item := map[string]any{
			"id": name, "object": "model", "created": created, "owned_by": "ollama",
		}
		if name == listed {
			item["default"] = true
		}
		data = append(data, item)
	}
	return map[string]any{"object": "list", "default": listed, "data": data}, nil
}

func unknownModelMessage(requested string, names []string) string {
	if len(names) == 0 {
		return fmt.Sprintf("model %q is not in Ollama's model list; Ollama has no models", requested)
	}
	return fmt.Sprintf("model %q is not in Ollama's model list; available: %s", requested, strings.Join(names, ", "))
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var chatReq ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		log.Printf("ERROR: failed to decode chat request: %s", err.Error())
		return
	}
	if chatReq.Model == "" {
		chatReq.Model = activeConfig.model
	}
	names, err := ollamaModelNames()
	if err != nil {
		log.Printf("ERROR: %s", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	if !modelInList(chatReq.Model, names) {
		msg := unknownModelMessage(chatReq.Model, names)
		log.Printf("ERROR: %s", msg)
		http.Error(w, msg, http.StatusNotFound)
		return
	}
	messages, err := toOllamaMessages(chatReq.Messages)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	ollamaReq := OllamaRequest{Model: chatReq.Model, Messages: messages, Stream: chatReq.Stream, Tools: chatReq.Tools}
	if chatReq.Temperature != nil || chatReq.MaxTokens != nil {
		ollamaReq.Options = map[string]any{}
		if chatReq.Temperature != nil {
			ollamaReq.Options["temperature"] = *chatReq.Temperature
		}
		if chatReq.MaxTokens != nil {
			ollamaReq.Options["num_predict"] = *chatReq.MaxTokens
		}
	}
	// Ollama's native /api/chat API supports tools but does not expose OpenAI's
	// tool_choice semantics consistently. Let Ollama select tools automatically.
	if len(ollamaReq.Tools) == 0 {
		for _, fn := range chatReq.Functions {
			ollamaReq.Tools = append(ollamaReq.Tools, Tool{Type: "function", Function: fn})
		}
	}
	body, err := json.Marshal(ollamaReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	ollamaResp, err := http.Post(fmt.Sprintf("%s/chat", activeConfig.endpoint), "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Printf("ERROR: POST to Ollama failed: %s", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer ollamaResp.Body.Close()
	if ollamaResp.StatusCode < 200 || ollamaResp.StatusCode >= 300 {
		errBody, _ := io.ReadAll(ollamaResp.Body)
		http.Error(w, fmt.Sprintf("Ollama returned %s: %s", ollamaResp.Status, errBody), ollamaResp.StatusCode)
		return
	}
	content, native, err := readOllamaChat(ollamaResp.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	tools := catalogFromTools(ollamaReq.Tools)
	text, calls := resolveAssistantOutput(content, native, tools)
	if len(native) == 0 && len(calls) > 0 {
		names := make([]string, len(calls))
		for i, call := range calls {
			names[i] = call.Function.Name
		}
		log.Printf("Translated JSON content into tool call(s): %s", strings.Join(names, ", "))
	}
	writeCompletion(w, chatReq.Model, text, calls, chatReq.Stream)
}

func writeSSE(w http.ResponseWriter, flusher http.Flusher, payload map[string]any) {
	if data, err := json.Marshal(payload); err == nil {
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func writeCompletion(w http.ResponseWriter, model, text string, calls []OllamaToolCall, stream bool) {
	created := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	openAICalls := toOpenAIToolCalls(calls, created)
	if !stream {
		var content any = text
		if text == "" && len(calls) > 0 {
			content = nil
		}
		message := map[string]any{"role": "assistant", "content": content}
		reason := "stop"
		if len(openAICalls) > 0 {
			for _, call := range openAICalls {
				delete(call, "index")
			}
			message["tool_calls"] = openAICalls
			reason = "tool_calls"
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "message": message, "finish_reason": reason}},
		})
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	chunk := func(delta map[string]any, finish any) map[string]any {
		return map[string]any{
			"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
		}
	}
	sent := false
	if strings.TrimSpace(text) != "" {
		writeSSE(w, flusher, chunk(map[string]any{"role": "assistant", "content": text}, nil))
		sent = true
	}
	if len(openAICalls) > 0 {
		delta := map[string]any{"tool_calls": openAICalls}
		if !sent {
			delta["role"] = "assistant"
			delta["content"] = nil
		}
		writeSSE(w, flusher, chunk(delta, nil))
		writeSSE(w, flusher, chunk(map[string]any{}, "tool_calls"))
	} else {
		delta := map[string]any{}
		if !sent {
			delta["role"] = "assistant"
			delta["content"] = ""
		}
		writeSSE(w, flusher, chunk(delta, "stop"))
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func handleModelsRequest(w http.ResponseWriter) {
	names, err := ollamaModelNames()
	if err != nil {
		log.Printf("ERROR: %s", err)
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	payload, err := modelsList(names, activeConfig.model)
	if err != nil {
		log.Printf("ERROR: %s", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(payload)
}
