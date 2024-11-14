import streamlit as st
from upload_arcticles import find_similar_articles


# Helper function to handle the user input selection
def handle_user_input(input_type):
    if input_type == "PDF File":
        user_input = st.file_uploader("Upload a PDF file", type="pdf")
        if user_input:
            file_path = save_uploaded_file(user_input)
            return file_path

    elif input_type == "Image File (PNG/JPG/JPEG)":
        user_input = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
        if user_input:
            file_path = save_uploaded_file(user_input)
            return file_path

    elif input_type == "arXiv ID":
        user_input = st.text_input("Enter the arXiv ID:")
        if user_input:
            return user_input

    elif input_type == "arXiv.org Link":
        user_input = st.text_input("Enter the arXiv.org link:")
        if user_input:
            return user_input

    return None


# Function to save the uploaded file and return the path
def save_uploaded_file(uploaded_file):
    with open(f"temp/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return f"temp/{uploaded_file.name}"


# Streamlit UI
def search_by_paper():
    st.title("🔍 Search for Similar Articles")

    # Dropdown menu to select input type
    input_type = st.selectbox(
        "Select the input type:",
        ["PDF File", "Image File (PNG/JPG/JPEG)", "arXiv ID", "arXiv.org Link"]
    )

    # Handle user input based on the selected input type
    user_input = handle_user_input(input_type)

    # Submit button to trigger the search
    if st.button("Submit"):
        if user_input:
            st.write("### Processing your input...")
            similar_articles = find_similar_articles(user_input)

            if similar_articles:
                st.write("## Similar Articles Found:")
                for idx, article in enumerate(similar_articles, start=1):
                    title = article['title']
                    abstract_summary = article['abstract']

                    # Display each article's title and abstract summary
                    st.markdown(f"""
                        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                            <h3 style="color: #2c3e50;">{title}</h3>
                            <p style="color: #34495e;">{abstract_summary}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No similar articles found or an issue occurred during processing.")
        else:
            st.error("Please provide a valid input.")