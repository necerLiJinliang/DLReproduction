lm_eval --model hf \
    --model_args pretrained=model_save/base_model/qwen-1.5-1.8b,parallelize=True,load_in_4bit=True,peft=model_save/lambda_dpo_model/qwen-1.5-1.8b-dpo-lambda \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 