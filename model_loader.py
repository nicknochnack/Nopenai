import torch
import transformers
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
print(torch.cuda.is_available())
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

name = 'mosaicml/mpt-7b-instruct'
#name = 'mosaicml/mpt-7b'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!
config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model.eval()
model.to(device)
print(f"Model loaded on {device}")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

example = "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week? Explain before answering."
fmt_ex = PROMPT_FOR_GENERATION_FORMAT.format(instruction=example)

# mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=64,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])

print("Now using LLM chain and HF pipeline")
# template for an instruction with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}"
)

llm = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=llm, prompt=prompt)

print(llm_chain.predict(
    instruction="Explain to me the difference between nuclear fission and fusion."
).lstrip())