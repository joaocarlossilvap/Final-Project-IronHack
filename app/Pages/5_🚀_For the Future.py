import streamlit as st
from streamlit_extras.colored_header import colored_header

with st.sidebar: 
    st.image('https://seeklogo.com/images/I/ironhack-logo-F751CF4738-seeklogo.com.png')
    st.title('LipNet Project')

colored_header(
    label="For The Future",
    description="The Dream",
    color_name="violet-70",
)

st.subheader("I would love to contribute to LipNet's development so that people like Emanuel could read lips through their phones or computers during live calls or using their cameras.")

st.subheader("This way, he could overcome his communication difficulties and have a more accessible and fulfilling interaction with the world.")