quadruplet_instruction = """Hãy xác định loại khía cạnh, cụm từ thể hiện khía cạnh, cụm từ thể hiện ý kiến và trạng thái ý kiến (tốt tạm, tệ) cho bình luận sau đây: \"{}\"\nTrả lời:"""

def get_prompt(input_review, task):
    if task == 'quadruplet':
        instruction = quadruplet_instruction.format(input_review)
    
    prompt = instruction.format(input_review)
    return prompt