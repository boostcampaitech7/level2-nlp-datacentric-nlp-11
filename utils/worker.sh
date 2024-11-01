#!/bin/bash

QUEUE_FILE="$(pwd)/job_queue.txt"
PROCESS_RUNNING=false

# Ctrl+C로 작업을 멈추고 다음 작업으로 이동하게 하는 핸들러 설정
trap 'echo "작업을 건너뜁니다..."; PROCESS_RUNNING=false' SIGINT

while true; do
    if [ -s "$QUEUE_FILE" ]; then
        # 첫 번째 줄 읽기
        JOB=$(head -n 1 "$QUEUE_FILE")

        # 명령어 실행
        echo "작업을 시작합니다: $JOB"
        PROCESS_RUNNING=true
        eval "$JOB" &
        JOB_PID=$!

        # 작업 진행 중 Ctrl+C 입력을 기다림
        wait $JOB_PID || true
        PROCESS_RUNNING=false

        # 첫 번째 줄 삭제
        sed -i '1d' "$QUEUE_FILE"
    fi

    # 짧은 대기 후 다시 확인
    sleep 2
done
