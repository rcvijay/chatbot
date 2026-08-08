import streamlit as st
import requests

st.set_page_config(page_title="Internal Policy Assistant", page_icon="📑", layout="wide")

st.title("📑 Internal Policy & HR Assistant")
st.markdown("Ask questions about company policies, travel reimbursements, leave limits, or IT requests.")

BACKEND_URL = "http://nginx/api/query"

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📄 View Cited Documents"):
                for cite in message["citations"]:
                    st.write(f"- **{cite['doc_name']}** ({cite['section']}) - *Date: {cite['effective_date']}*")

if prompt := st.chat_input("e.g., How many casual leaves carry forward?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"question": prompt},
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    
                    st.markdown(answer)
                    if citations:
                        with st.expander("📄 View Cited Documents"):
                            for cite in citations:
                                st.write(f"- **{cite['doc_name']}** ({cite['section']}) - *Date: {cite['effective_date']}*")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "citations": citations
                    })
                else:
                    st.error("Error communicating with the backend server.")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")