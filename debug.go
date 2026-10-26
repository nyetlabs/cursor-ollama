package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

const debugLogName = "cursor-ollama-debug.log"

var (
	debugMu   sync.Mutex
	debugFile *os.File
)

func debugFlagSet() bool {
	for _, arg := range os.Args[1:] {
		if arg == "--debug" || arg == "-debug" {
			return true
		}
	}
	return false
}

func openDebugLog() error {
	file, err := os.OpenFile(debugLogName, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	debugFile = file
	return nil
}

type captureWriter struct {
	header http.Header
	code   int
	body   bytes.Buffer
}

func (c *captureWriter) Header() http.Header {
	if c.header == nil {
		c.header = make(http.Header)
	}
	return c.header
}

func (c *captureWriter) WriteHeader(status int) {
	if c.code == 0 {
		c.code = status
	}
}

func (c *captureWriter) Write(b []byte) (int, error) {
	if c.code == 0 {
		c.code = http.StatusOK
	}
	return c.body.Write(b)
}

func (c *captureWriter) Flush() {}

func debugEntry(method, path string, reqBody []byte, status int, respBody []byte) string {
	if status == 0 {
		status = http.StatusOK
	}
	return fmt.Sprintf("--- %s %s %s\nREQUEST\n%s\nRESPONSE %d\n%s\n\n",
		time.Now().Format(time.RFC3339Nano), method, path, bytes.TrimSpace(reqBody), status, bytes.TrimSpace(respBody))
}

func writeDebugEntry(method, path string, reqBody []byte, status int, respBody []byte) {
	if debugFile == nil {
		return
	}
	debugMu.Lock()
	defer debugMu.Unlock()
	fmt.Fprint(debugFile, debugEntry(method, path, reqBody, status, respBody))
}

func (c *captureWriter) replay(w http.ResponseWriter) {
	code := c.code
	if code == 0 {
		code = http.StatusOK
	}
	for key, values := range c.header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(code)
	w.Write(c.body.Bytes())
}

// serveDebug logs the response serveProxy would return. Chat completions are
// answered with a finished "ok" message so Cursor does not retry. Other
// routes, including /v1/models, are passed through.
func serveDebug(w http.ResponseWriter, r *http.Request) bool {
	if !activeConfig.debug {
		return false
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return true
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	captured := &captureWriter{}
	serveProxy(captured, r)
	writeDebugEntry(r.Method, r.URL.RequestURI(), body, captured.code, captured.body.Bytes())
	if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
		captured.replay(w)
		return true
	}
	var req struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	_ = json.Unmarshal(body, &req)
	if req.Model == "" {
		req.Model = activeConfig.model
	}
	enableCors(w)
	writeCompletion(w, req.Model, "ok", nil, req.Stream)
	return true
}
