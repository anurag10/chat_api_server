import logging
from transformers import logging as hf_logging
from transformers import AutoTokenizer, AutoModelForCausalLM

class chat_completion:
    def __init__(self):
        # Set Hugging Face transformers logging to ERROR (or CRITICAL to suppress most messages)
        hf_logging.set_verbosity_error() 
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", local_files_only=True)
        # self.few_shot_examples = """
        #     User: What's 5 + 3?
        #     Bot: 5 + 3 equals 8.
        #     User: Tell me a joke?
        #     Bot: Why don't some couples go to the gym? Because some relationships don't work out!.
        # """
        # self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", local_files_only=True)
        # self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", local_files_only=True)

    def get_model_response(self, input_text):
        # user_query = self.few_shot_examples + "\n" + f"User: {input_text}\nBot:"
        # print(user_query)
        input_ids = self.tokenizer(input_text, return_tensors="pt")

        print("tokenizer output S")
        print(input_ids)
        # outputs = model.generate(**input_ids)
        outputs = self.model.generate(
            **input_ids,
            num_beams=3, #use beam search
            num_return_sequences=1,
            max_new_tokens=500,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            early_stopping=True)
        print("model output ")
        print(outputs)
        tokenizer_output = (self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        tokenizer_output = tokenizer_output.replace(input_text, '').strip()
        return tokenizer_output

#### use quantized model 2b
# pip install bitsandbytes accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# # # 8 bit precision
# # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# # 4 bit precision
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config)

# input_text = "Tell me something interesting about fishes?"
# input_ids = tokenizer(input_text, return_tensors="pt").to("cpu")
# model.to("cpu")  # Move model to CPU

# output = model.generate(input_ids["input_ids"])
# print(tokenizer.decode(output[0], skip_special_tokens=True))
# print(tokenizer.decode(outputs[0]))