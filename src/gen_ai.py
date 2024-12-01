from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch

prompt_template = PromptTemplate(
    template="""Given the context: {context}, answer the question: {question}.""",
    input_variables=["context", "question"])

def generate_answer(user_query, retrieved_context):
    # model_name = "facebook/opt-1.3b"
    # model_name = "tiiuae/falcon-11B"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # model_name = "openai-community/gpt2-large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(device)

    llm_pipeline = pipeline(
        "text-generation", 
        model=AutoModelForCausalLM.from_pretrained(model_name).to(device), 
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.1,
        top_k=50,
        repetition_penalty=1.1,
        # device=0
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    
    formatted_prompt = prompt_template.format(context=retrieved_context, question=user_query)
    
    return llm(formatted_prompt)