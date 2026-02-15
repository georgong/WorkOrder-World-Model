# python -m src.runner.train_kfold --data_path data/graph/sdge.pt --target assignments --model sage --k 2 --seed 42 --save_best
# python -m src.runner.train_kfold --data_path data/graph/sdge.pt --target assignments --model hgt --k 2 --seed 42 --save_best
python -m src.runner.train_kfold --data_path data/graph/sdge.pt --target assignments --model rgcn --k 2 --seed 42 --save_best