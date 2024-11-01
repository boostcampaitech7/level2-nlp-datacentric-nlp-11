#!/bin/bash
SOUND_FILE="$(pwd)/notify.wav "
QUEUE_FILE="$(pwd)/job_queue.txt"

while true; do
    if [ -s "$QUEUE_FILE" ]; then
        # 첫 번째 줄 읽기
        JOB=$(head -n 1 "$QUEUE_FILE")

        # 명령어 실행
        eval "$JOB"

        # 첫 번째 줄 삭제
        sed -i '1d' "$QUEUE_FILE"
        aplay "$SOUND_FILE"
    fi
    # 짧은 대기 후 다시 확인
    sleep 2
done
