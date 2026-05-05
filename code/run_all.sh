#!/bin/bash
set -e

echo "=== CoLight Jinan ==="
python run_colight.py --dataset jinan -memo ncolight_jinan -workers 3

echo "=== Advanced MPLight Hangzhou ==="
python run_advanced_mplight.py --dataset hangzhou -memo adv_mplight_hangzhou -workers 3

echo "=== Advanced MPLight Xuancheng (20 intersections) ==="
python run_advanced_mplight.py --dataset xuancheng_mod -memo adv_mplight_xuancheng -workers 1

echo "=== ALL DONE ==="
