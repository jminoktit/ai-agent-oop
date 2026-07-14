import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="AuraBook Assistant", page_icon="🎓")

st.title("🎓 AuraBook Assistant")
st.markdown("AI University Assistant - Programming, Math, Education")

@st.cache_resource
def load_model():
    model_name = "YOUR_USERNAME/aura-book-model"  # غيّر لاسم الموديل بتاعك
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

pipe = load_model()

SYSTEM_PROMPT = (
    "You are Aura, a helpful university assistant specializing in programming, "
    "mathematics, and education. You speak Arabic and English fluently."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اسألني أي سؤال..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    formatted = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = pipe(formatted, max_new_tokens=512, temperature=0.7, do_sample=True)
    resp = output[0]["generated_text"]

    if "<|assistant|>" in resp:
        resp = resp.split("<|assistant|>")[-1]
    resp = resp.replace("</s>", "").strip()

    st.session_state.messages.append({"role": "assistant", "content": resp})
    with st.chat_message("assistant"):
        st.markdown(resp)
