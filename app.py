import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_text(prompt, model, tokenizer, max_length=5000):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text.strip()

def check_text(text):
    # Generate text using the model
    generated_text = generate_text(text, model, tokenizer)

    # Analyze the response and classify as AI-generated or human-written
    if generated_text.lower() == text.lower():
        return "This text is likely AI-generated."
    else:
        return "This text appears to be human-written."

def main():
    st.title("Text Detector")

    # Get user input
    user_input = st.text_area("Enter text:")

    if st.button("Check"):
        if user_input:
            result = check_text(user_input)
            st.write(result)
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()

