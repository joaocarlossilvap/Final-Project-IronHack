import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.mention import mention

with st.sidebar: 
    st.image('https://seeklogo.com/images/I/ironhack-logo-F751CF4738-seeklogo.com.png')
    st.title('LipNet Project')

colored_header(
    label="Thank You",
    description="Conclusion",
    color_name="violet-70",
)

def main():
    gif_path = "/Users/joaopereira/Documents/Ir0nH@ck/Projects/Final Project/LipNet-main/app/Pages/readlipsgif.gif"
    st.image(gif_path, use_column_width=True)

if __name__ == "__main__":
    main()  