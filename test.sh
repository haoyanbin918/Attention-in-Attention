python test_models.py somethingv1 RGB --arch 'resnet50' --gpu='0' \
    --test_nets 'C2D_CinST_BN' --test_weights=checkpoint_240320/C2D_CinST_BN_resnet50_somethingv1_RGB_avg_segment8_e50_s20p40_twice_ef_drp5_lr01/ckpt_test.best.pth.tar \
    --test_segments 8 \
    --test_alphas 1 --test_betas 1 \
    --element_filter --cdiv 4 \
    --batch-size 1 -j 0 --dropout 0.5 --consensus_type=avg \
    --full_res --test_crops 1 