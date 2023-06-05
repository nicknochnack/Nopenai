from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch

import torch

print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

template = """{question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])




model_id = 'microsoft/CodeGPT-small-java-adaptedGPT2'
model_id = 'microsoft/CodeGPT-small-java'
model_id = 'codeparrot/codeparrot-small'
model_id = 'mosaicml/mpt-7b-instruct'
model_id = "codeparrot/codeparrot-small-multi"


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=True)
#config.attn_config['attn_impl'] = 'triton'
#config.init_device = 'cuda:0' # For fast initialization directly on GPU!
config.init_device = 'cpu' # For fast initialization directly on GPU!
config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096
model = transformers.AutoModelForCausalLM.from_pretrained(
  model_id,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)
device = 'cpu'
model.eval()
model.to(device)
print(f"Model loaded on {device}")

pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=300
)
local_llm = HuggingFacePipeline(pipeline=pipeline)

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = """def is_even(value):
    Returns True if value is an even number.
    return value % 2 == 0

    # setup unit tests for is_even
    import unittest
"""

print(llm_chain.run(question))