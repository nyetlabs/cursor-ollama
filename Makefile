.PHONY: all test run

all:
	mkdir -p build
	CGO_ENABLED=0 go build -o build/cursor-ollama .

test:
	go test ./...

run: all
	./build/cursor-ollama

debug: all
	./build/cursor-ollama --debug