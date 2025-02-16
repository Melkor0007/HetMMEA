CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python  main.py \
            --gpu           $1    \
            --eval_epoch    1  \
            --only_test     0   \
            --model_name    MCLEA \
            --data_choice   $2 \
            --data_split    $3 \
            --data_rate     $4 \
            --epoch         500 \
            --lr            5e-4  \
            --hidden_units  "600,600,600" \
            --save_model    0 \
            --batch_size    300 \
            --semi_learn_step 5 \
	        --csls          \
	        --csls_k        3 \
	        --random_seed   $5 \
            --exp_name      MCLEA_MLLM_v${7}\
            --exp_id        ver_${7} \
            --workers       12 \
            --accumulation_steps 1 \
            --scheduler     cos \
            --attr_dim      600     \
            --img_dim       600     \
            --name_dim      600     \
            --char_dim      600     \
            --hidden_size   300     \
            --tau           0.1     \
            --tau2          4.0     \
            --structure_encoder "gat" \
            --num_attention_heads 1 \
            --num_hidden_layers 1 \
            --use_surface   $6     \
            --add_noise 0 \
            --lr_change_epoch 400\
            # --il            \
	        # --il_start      250 \
            # --unsup \
            # --unsup_mode $8 \
            # --unsup_k 3000  \
