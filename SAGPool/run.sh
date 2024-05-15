# python3 main.py --ablation 0, 1, 2
#    0: baseline
#    1: cycle with feature = 0
#    2: cycle with feature = mode value

# python3 main.py --ablation 0
# python3 main.py --ablation 1

for i in {0..1}
    for j in {1..30}
    do
        python3 main.py --ablation $i
    done
done