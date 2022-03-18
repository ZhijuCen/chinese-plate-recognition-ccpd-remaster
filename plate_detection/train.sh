
python main_trainer.py --dataset "../linked_datasets/CCPD2019" "../linked_datasets/CCPD2020" \
--train-splits "../third_party/CCPD/split/train.txt" "../third_party/CCPD/split/green_train.txt" \
--train-mapping 0 1 \
--val-splits "../third_party/CCPD/split/val.txt" "../third_party/CCPD/split/green_val.txt" \
--val-mapping 0 1
