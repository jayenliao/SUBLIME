cd ..

python3 main.py -dataset UR -ntrials 2 -sparse 1 -epochs_cls 200 -lr_cls 0.001 -w_decay_cls 0.0005 -hidden_dim_cls 32 -dropout_cls 0.5 -dropedge_cls 0.75 -nlayers_cls 2 -patience_cls 10 -epochs 100 -lr 0.01 -w_decay 0.0 -hidden_dim 512 -rep_dim 256 -proj_dim 256 -dropout 0.5 -dropedge_rate 0.5 -nlayers 2 -type_learner att -k 30 -sim_function cosine -activation_learner relu -gsl_mode structure_refinement -eval_freq 10 -tau 0.9999 -maskfeat_rate_learner 0.7 -maskfeat_rate_anchor 0.6 -contrast_batch_size 0 -c 0 -device cuda:0 -dataset_type pure -out_models gcn
