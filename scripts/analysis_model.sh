python -m src.runner.compare_model \
  --pt data/graph/sdge.pt \
  --sage_ckpt path/to/sage.pt
  --mlp_ckpt path/to/mlp.pt \
  --lgb_ckpt path/to/lightgbm.txt \
  --split test \
  --out runs/compare_model/compare_three.png \
  --predictions_out runs/compare_model/predictions.json


python -m src.runner.analysis_model --predictions runs/compare_model/predictions.json --plotly