from transformers import pipeline
from googletrans import Translator

translator = Translator()
qa_pipeline = pipeline("question-answering")

def answer_question(context, question, language):
    # Translate question to English if needed
    if language != "English":
        lang_code = "hi" if language == "Hindi" else "te"
        question = translator.translate(question, dest="en").text
    else:
        lang_code = "en"

    result = qa_pipeline(question=question, context=context)
    answer = result["answer"]

    # Translate answer back to selected language
    if language != "English":
        answer = translator.translate(answer, dest=lang_code).text

    return answer