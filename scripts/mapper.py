import os



education_dict = {
    'Behavior': 'hành vi',              # Behavior
    'Curriculum': 'chương trình giảng dạy', # Curriculum
    'Equipment': 'thiết bị',            # Equipment
    'Exercise': 'bài tập',              # Exercise
    'Experience': 'kinh nghiệm',        # Experience
    'General': 'vấn đề chung',                 # General
    'Grading': 'chấm điểm',             # Grading
    'Knowledge': 'kiến thức',           # Knowledge
    'Lecture Material': 'tài liệu giảng dạy', # Lecture Material
    'Suggestion': 'đề xuất',            # Suggestion
    'Teaching Skill': 'kỹ năng giảng dạy' # Teaching Skill
}



def mapping_category(domain, category, lang_return='eng') 
    if domain == "Education":
        mapped_dict = education_dict  
    else:
        print("ERRORRR IN mapping_category FUNCTION: , ", domain)
        return None
    
    inversed_dict = {viet: eng for eng, viet in mapped_dict.items()}

    
    if lang_return == 'vie':
        return mapped_dict[category]
    else:
        return inversed_dict[category]


SENTIMENT_VIET2ENG = {
    'tốt': 'positive',
    'tệ': 'negative',
    'tạm': 'neutral'
}

SENTIMENT_ENG2VIET = {
    'positive': 'tốt',
    'negative': 'tệ',
    'neutral': 'tạm'
}