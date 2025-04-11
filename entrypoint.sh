#!/bin/sh
set -e

ollama serve &

max_attempts=30
attempt_num=1
until curl -s http://localhost:11434 >/dev/null; do
    if [ $attempt_num -ge $max_attempts ]; then
        echo "Max attempts reached, ollama service did not start"
        exit 1
    fi
    echo "Waiting for ollama service to start..."
    sleep 1
    attempt_num=$((attempt_num+1))
done

wait
