PROMPT_2 = """[INST] <<SYS>>
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<</SYS>>

{} [/INST]"""

PROMPT_SEALLM_V2 = """\
<s><|im_start|>system
You are a helpful assistant.</s><|im_start|>user
{}</s><|im_start|>assistant
"""

PROMPT_SEALLM_V2_5 = """\
<bos><|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.<eos>
<|im_start|>user
{}<eos>
<|im_start|>assistant
"""

PROMPT_SEALLM_V3 = """\
<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""


pair_instruction = """Hãy xác định loại khía cạnh và trạng thái ý kiến (tốt, tạm, tệ) cho bình luận sau đây: \"{}\"\nTrả lời:"""

triplet_instruction = """Hãy xác định loại khía cạnh, cụm từ thể hiện khía cạnh và trạng thái ý kiến (tốt, tạm, tệ) cho bình luận sau đây: \"{}\"\nTrả lời:"""

quadruplet_instruction = """Hãy xác định loại khía cạnh, cụm từ thể hiện khía cạnh, cụm từ thể hiện ý kiến và trạng thái ý kiến (tốt tạm, tệ) cho bình luận sau đây: \"{}\"\nTrả lời:"""

def get_prompt(input_review, prompt_format, task):

    if prompt_format == 'SEALLM_V2':
        prompt = PROMPT_SEALLM_V2
    elif prompt_format == 'SEALLM_V2_5':
        prompt = PROMPT_SEALLM_V2_5
    elif prompt_format == 'SEALLM_V3':
        prompt = PROMPT_SEALLM_V3
    else:
        prompt = None

    if task == 'pair':
        instruction = pair_instruction.format(input_review)
    elif task == 'triplet':
        instruction = triplet_instruction.format(input_review)
    elif task == 'quadruplet':
        instruction = quadruplet_instruction.format(input_review)
    
    if prompt:
        prompt = prompt.format(instruction.format(input_review))
    else:
        prompt = instruction.format(input_review)
    return prompt