# python3 main.py --ablation 0, 1, 2
#    0: baseline
#    1: cycle with feature = 0
#    2: cycle with feature = mode value

# python3 main.py --ablation 0
# python3 main.py --ablation 1

# for i in $(seq 3 3)
# do
#     for j in $(seq 1 30)
#     do
#         python3 main.py --dataset PROTEINS --ablation $i
#     done
# done

for j in $(seq 1 30)
do
    python3 main.py --dataset PROTEINS --ablation 2
done