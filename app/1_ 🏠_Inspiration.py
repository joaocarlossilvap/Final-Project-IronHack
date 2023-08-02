import streamlit as st
from PIL import Image
from streamlit_extras.colored_header import colored_header


st.set_page_config(
    page_title = "LipRead @ IronHack",
    page_icon = "💬" 
    )

with st.sidebar: 
    st.image('https://seeklogo.com/images/I/ironhack-logo-F751CF4738-seeklogo.com.png')
    st.title('LipNet Project')

colored_header(
    label="Introduction",
    description="The Inspiration",
    color_name="violet-70",
)
image = Image.open('/Users/joaopereira/Documents/Ir0nH@ck/Projects/Final Project/LipNet-main/app/emanuel.jpg')
st.image(image, caption='Emanuel Gonçalves')

def main():

    st.markdown("""
    • Emanuel: Inspiring friend; deaf since birth. \n\n
    • Communication Challenge: Relies on lip reading. \n\n
    • Dream: National winner in women's futsal sports as coach. \n\n
    • Determination: No deafness obstacle; immense dedication. \n\n
    • Seeking Solutions: Support better communication. \n\n
    • LipNet: Improves lip reading accuracy. \n\n
    • Path to Accessibility: More accessible communication. \n\n
    • Empowering Dreams: Overcoming barriers; pursuing dreams. \n\n
    • Building a Supportive World: Inclusive tech for hearing challenges. \n\n
    """)

if __name__ == "__main__":
    main()




