import streamlit as st
import requests
from streamlit_extras.colored_header import colored_header
from streamlit_lottie import st_lottie

with st.sidebar: 
    st.image('https://seeklogo.com/images/I/ironhack-logo-F751CF4738-seeklogo.com.png')
    st.title('LipNet Project')

colored_header(
    label="How it Works?",
    description="Tools",
    color_name="violet-70",
)

with st.expander("Deep Learning Frameworks like TensorFlow, PyTorch, and Keras"):
         st.write("Specialized tools that enable developers to build smart computer programs that can learn and understand complex patterns.")                  

with st.expander("Convolutional Neural Networks (CNNs)"):
         st.write("Use filters to find patterns and make the video simpler. Then, CNN saves the important parts to keep the main info to train te model.")                  

with st.expander("Long Short-Term Memory (LSTM)"):
        st.write("A specialized program that helps computers remember and understand patterns that evolve over time, like predicting the next word in a sentence.")

with st.expander("OpenCV to Data Collection and Preprocessing"):
        st.write("Utilizes OpenCV, a library of tools, to collect and prepare image data for computer vision tasks.")

with st.expander("Data Augmentation Libraries"):
        st.write("Tools that generate diverse data variations to improve machine learning models' adaptability.")

colored_header(
    label="But How it Works?",
    description="Explanation",
    color_name="violet-70",
)   

def main():

    st.markdown("""
    • LipNet uses a video of someone speaking and an align file with word timings. \n\n
    • It analyzes the video using CNNs to study lip movements. \n\n
    • Uses a PipeLine to process data, learn from it, and make predictions. \n\n
    • Another network (LSTM) helps understand how the lips move over time. \n\n 
    • By matching the lip movements to the align file, LipNet transcribes the spoken words. \n\n
    • This allows accurate lip-reading and speech recognition. \n\n
    """)

if __name__ == "__main__":
    main()