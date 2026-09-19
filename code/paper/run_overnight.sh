#!/bin/bash
cd "/home/bibesh/Desktop/MultiModel LLM's"
source venv/bin/activate

echo "=== START: $(date) ==="

python3 code/run_verification.py --model internvl3 --prompting zero_shot        --all-impostors --run &&
python3 code/run_verification.py --model internvl3 --prompting task_description --all-impostors --run &&
python3 code/run_verification.py --model internvl3 --prompting similarity_score --all-impostors --run &&

python3 code/run_verification.py --model qwen25vl  --prompting zero_shot        --all-impostors --run &&
python3 code/run_verification.py --model qwen25vl  --prompting task_description --all-impostors --run &&
python3 code/run_verification.py --model qwen25vl  --prompting similarity_score --all-impostors --run &&

echo "=== ALL DONE: $(date) ==="
