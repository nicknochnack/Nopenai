from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

template = """{question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])


model_id = "codeparrot/codeparrot-small-multi"
model_id = 'mosaicml/mpt-7b-instruct'

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

pipeline = pipeline(
    "text-generation", 
    model=model_id, 
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