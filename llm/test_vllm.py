from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, 
                                 top_k=-1,
                                 max_tokens=7, 
                                 ignore_eos=True,
                                 best_of=4,
                                 use_beam_search=True,
                                 n=3
                                 
                                 )

# Create an LLM.
llm = LLM(model="/mnt/c/Users/dufei/codes/ai/data/model/facebook-opt/350m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

