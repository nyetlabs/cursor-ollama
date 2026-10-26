package main

import (
	"net/http/httptest"
	"strings"
	"testing"
)

type flushRecorder struct {
	*httptest.ResponseRecorder
}

func (flushRecorder) Flush() {}

func TestDebugEntryRecordsRequestAndResponse(t *testing.T) {
	entry := debugEntry("POST", "/v1/chat/completions", []byte(`{"model":"qwen2.5-coder:14b"}`), 200, []byte(`{"object":"chat.completion"}`))
	if !strings.Contains(entry, "POST /v1/chat/completions") || !strings.Contains(entry, `{"model":"qwen2.5-coder:14b"}`) {
		t.Fatalf("request missing from log entry: %s", entry)
	}
	if !strings.Contains(entry, "RESPONSE 200") || !strings.Contains(entry, `{"object":"chat.completion"}`) {
		t.Fatalf("response missing from log entry: %s", entry)
	}
}

func TestDebugOKCompletionStops(t *testing.T) {
	for _, stream := range []bool{false, true} {
		rec := flushRecorder{ResponseRecorder: httptest.NewRecorder()}
		writeCompletion(rec, "gpt-oss:20b", "ok", nil, stream)
		body := rec.Body.String()
		if !strings.Contains(body, `"content":"ok"`) && !strings.Contains(body, `"content": "ok"`) {
			t.Fatalf("stream=%v missing ok content: %s", stream, body)
		}
		if !strings.Contains(body, `"finish_reason":"stop"`) && !strings.Contains(body, `"finish_reason": "stop"`) {
			t.Fatalf("stream=%v missing stop: %s", stream, body)
		}
		if strings.Contains(body, "tool_calls") {
			t.Fatalf("stream=%v included a tool call: %s", stream, body)
		}
		if stream && !strings.Contains(body, "data: [DONE]") {
			t.Fatalf("stream missing done marker: %s", body)
		}
	}
}
