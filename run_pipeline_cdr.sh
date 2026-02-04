#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate /home/labs/schreiber/juliav/tools/SiteAF3/env

cd /home/labs/schreiber/juliav/tools/SiteAF3

python -u run_SiteAF3.py \
    --model_weights_dir ../../models/ \
    --db_dir /shareDB/alphafold3/ \
    --receptor_type protein \
    --ligand_type protein \
    --use_af3_msa_for_embedding \
    --use_pocket_msa_for_embedding \
    --use_pocket_masked_af3_msa_for_embedding \
    --cdr_noise_spread 3.0 \
    --framework_noise_spread 10.0 \
    --verbose \
    --config_file rbd_7vnb-ab-101021_data_siteaf3_nomsa_cdr.json \
    --output_dir sample_output_cdr \
    > /home/labs/schreiber/juliav/tools/SiteAF3/log_cdr 2>&1
