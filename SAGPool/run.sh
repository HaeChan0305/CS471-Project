# python3 main.py --ablation 0, 1, 2
#    0: baseline
#    1: cycle with feature = 0
#    2: cycle with feature = mode value

# python3 main.py --ablation 0
# python3 main.py --ablation 1

# for i in $(seq 0 1)
# do
for j in $(seq 11 11)
do
    python main.py --file logs_ablation2_NCI1 --iter $j
done
# done
