import streamlit as st
from streamlit_extras.colored_header import colored_header

with st.sidebar: 
    st.image('https://seeklogo.com/images/I/ironhack-logo-F751CF4738-seeklogo.com.png')
    st.title('LipNet Project')

colored_header(
    label="What Is LipNet Architecture?",
    description="Description",
    color_name="violet-70",
)

def main():

    st.markdown("""
    • LipNet is a neural network that reads lips in videos and converts them into text. It can handle videos of different lengths and is trained as a complete system from start to finish.\n\n
    • And a deep learning architecture designed to perform visual speech recognition \n\n
    • LipNet focuses on decoding speech directly from the movements of the lips. \n\n
    • Uses Convolutional Neural Networks (CNNs) \n\n
    • Uses Long Short-Term Memory (LSTM) \n\n
    • Processes spatial and temporal information \n\n
    """)

if __name__ == "__main__":
    main()