import streamlit as st

def main():
    st.title(f"Healthcare Assistant")

    st.subheader(f":blue[Physiotherapy and Occupational Therapy]")
    st.write(f"**Insert your audio:**")
    left, middle, right = st.columns(3)
    if left.button("**Transcription 🏋️‍♀️**"):
        left.markdown("**Transcription**")
    if middle.button("**Summarization 🏋️‍♀️**"):
        middle.markdown("**Summarization**")

    st.subheader(f":blue[Medicine and Nursing]")
    st.write(f"**Insert your audio:**")
    left, middle1, middle2, right = st.columns(4)
    if left.button("**Transcription 🩺**"):
        left.markdown("**Transcription**")
    if middle1.button("**Summarization 🩺**"):
        middle1.markdown("**Summarization**")
    if middle2.button("**Classification 🩺**"):
        middle2.markdown("**Classification**")
    if right.button("**Severity 🩺**"):
        right.markdown("**Severity**")

    with st.sidebar:
        st.image("logo4.jpg")  # Adjust path if needed
        #st.title("**ROMTech**")
        #left, middle1, middle2, right = st.columns(4)
        if st.button("Manoj Sharma"):
            st.write("**Contact:** https://www.linkedin.com/in/manoj-sharma-b46b81aa/")
        if st.button("Omid Moridnejad"):
            st.write("**Contact:** https://www.linkedin.com/in/omid-moridnejad-2855a5151/")
        if st.button("Raquel Colares"):
            st.write("**Contact:** https://www.linkedin.com/in/raquel-colares-7b1327a0/")
        if st.button("Thuvaarakkesh Ramanathan"):
            st.write("**Contact:** https://www.linkedin.com/in/rt-rakesh/")

    
    
if __name__ == "__main__":
    st.sidebar.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #a0a4ac;
        }
    </style>
    """, unsafe_allow_html=True)
    main()