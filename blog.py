import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLaMA 2 model and tokenizer
@st.cache_resource()
def load_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change to a smaller model if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()
# Blog generation function
def generate_blog(prompt, max_length=500):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit UI
st.title("📝 AI Blog Generator (LLaMA 2)")
st.write("Generate high-quality blog posts using LLaMA 2!")

prompt = st.text_area("Enter a topic or starting sentence:")

if st.button("Generate Blog"):
    if prompt:
        with st.spinner("Generating..."):
            blog_content = generate_blog(prompt)
            st.subheader("📝 Generated Blog:")
            st.write(blog_content)
    else:
        st.warning("Please enter a prompt to generate content.")