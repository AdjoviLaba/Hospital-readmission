# 3. Prepare MIMIC-IV data
echo "Preparing MIMIC-IV data..."
python mimic4_dataset.py \
    --mimic_path $MIMIC_IV_PATH \
    --output_path $PROCESSED_DATA_DIR \
    --window $READMISSION_WINDOW \
    --seed $SEED \
    --test_size $TEST_SIZE

# 4. Run the readmission prediction model
echo "Running readmission prediction model..."
python run.py \
    --output_dir $MODEL_OUTPUT_DIR \
    --dropout $DROPOUT \
    --learning_rate $LEARNING_RATE \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --num_classes 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs $NUM_EPOCHS \
    --logging_steps $LOGGING_STEPS \
    --dataset_folder $PROCESSED_DATA_DIR \
    --multi_hop