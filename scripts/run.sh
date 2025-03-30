# Set environment variables
export HF_TOKEN='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'
export WANDB_DISABLED='true'

# Define common variables
NUM_EPOCHS=10
LORA_RANK=32
BATCH_SIZE=8
MODEL_TYPE=causal
GRADIENT_ACCUM_STEPS=2
MODEL_DIR="./models"

# Define a function to train a model with specified parameters
run_training() {
  local model_id=$1
  local domain=$2
  local prompt_format=$3

  python main.py --model_id "$model_id" \
                 --task pair \
                 --domain "$domain" \
                 --num_epochs "$NUM_EPOCHS" \
                 --lora_rank "$LORA_RANK" \
                 --batch_size "$BATCH_SIZE" \
                 --model_type "$MODEL_TYPE" \
                 --add_instruction \
                 --using_trainer \
                 --gradient_accumulation_steps "$GRADIENT_ACCUM_STEPS" \
                 ${prompt_format:+--prompt_format "$prompt_format"}
  
  # Remove model directory if it exists
  if [ -d "$MODEL_DIR" ]; then
    rm -r "$MODEL_DIR"
  fi
}

# Training configurations
# Restaurant domain
run_training "bigscience/bloomz-3b" "Restaurant"
run_training "SeaLLMs/SeaLLM-7B-v2" "Restaurant" "SEALLM_V2"
run_training "SeaLLMs/SeaLLM-7B-v2.5" "Restaurant" "SEALLM_V2_5"
run_training "SeaLLMs/SeaLLMs-v3-7B" "Restaurant" "SEALLM_V3"

# Phone domain
run_training "bigscience/bloomz-3b" "Phone"
run_training "SeaLLMs/SeaLLM-7B-v2" "Phone" "SEALLM_V2"
run_training "SeaLLMs/SeaLLM-7B-v2_5" "Phone" "SEALLM_V2_5"
run_training "SeaLLMs/SeaLLMs-v3-7B" "Phone" "SEALLM_V3"
