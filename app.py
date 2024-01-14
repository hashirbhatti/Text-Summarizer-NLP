import streamlit as st
import subprocess

# Set custom page configuration
st.set_page_config(page_title="SummarizeMe", page_icon="✍️", layout="wide")

def summarize_text(text):
    try:
        response = st.session_state.pipe.stdin.write(f"{text}\n")
        if response:
            summary = response
            return summary
        else:
            return f"Error Occurred! Unable to get summary from FastAPI backend."
    except Exception as e:
        return f"Error Occurred! {e}"

def start_fastapi_backend():
    subprocess.Popen(["uvicorn", "app_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

def main():
    st.title("Text Summarization App")

    text_input = st.text_area("Enter the text for summarization:", "")
    if st.button("Summarize"):
        if text_input:
            summary = summarize_text(text_input)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter text for summarization.")

if __name__ == "__main__":
    start_fastapi_backend()
    main()