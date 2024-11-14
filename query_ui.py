from rag import find_relevant_articles, refine_query_with_feedback
import streamlit as st


def fake_relevant_article():
    articles = [{'title': 'Article 1', 'summary': 'Summary of Article 1', 'abstract':"abstract 1", 'embedding': [0.1, 0.2, 0.3]},
               {'title': 'Article 2', 'summary': 'Summary of Article 2', 'abstract':"abstract 2", 'embedding': [0.2, 0.3, 0.4]},
                  {'title': 'Article 3', 'summary': 'Summary of Article 3', 'abstract':"abstract 3", 'embedding': [0.3, 0.4, 0.5]},
                {'title': 'Article 4', 'summary': 'Summary of Article 4', 'abstract':"abstract 4", 'embedding': [0.4, 0.5, 0.6]},
                {'title': 'Article 5', 'summary': 'Summary of Article 5', 'abstract':"abstract 5", 'embedding': [0.5, 0.6, 0.7]}]
    return articles


# Function to render the query search page
def query_search_page():
    st.title("🔍 Query Search Page")

    # Input field for user query
    user_input = st.text_input("Enter your query:", placeholder="Type your search query here...", key="user_query")

    # Search button
    if st.button("Search"):
        if user_input:
            # Fetch relevant articles and store them in session state
            st.session_state['similar_articles'] = find_relevant_articles(user_input)
            # st.session_state['similar_articles'] = fake_relevant_article()
            st.session_state['feedback'] = [None] * len(st.session_state['similar_articles'])

    # Check if there are articles stored in session state
    if 'similar_articles' in st.session_state:
        similar_articles = st.session_state['similar_articles']

        st.write("## Search Results:")
        for idx, article in enumerate(similar_articles, start=1):
            title = article['title']
            abstract_summary = article['summary']

            # Display each article's title and summary
            st.markdown(f"""
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                    <h3 style="color: #2c3e50;">{title}</h3>
                    <p style="color: #34495e;">{abstract_summary}</p>
                </div>
            """, unsafe_allow_html=True)

            # Add thumbs up and thumbs down radio buttons for feedback
            feedback_choice = st.radio(
                f"Is Article {idx} relevant?",
                ("👍 Yes", "👎 No"),
                key=f"feedback_{idx}",
                on_change=lambda idx=idx: update_feedback(idx)
            )

        # Refine button to process the feedback
        if st.button("Refine"):
            st.write("### Refining Search Results...")
            feedback = [1 if choice == "👍 Yes" else 0 for choice in st.session_state['feedback']]
            improved_articles = refine_query_with_feedback(
                article_embeddings=[article['embedding'] for article in similar_articles],
                feedback=feedback
            )

            # Display the refined articles
            st.write("## Refined Results:")
            for idx, article in enumerate(improved_articles, start=1):
                title = article['title']
                abstract_summary = article['summary']
                st.markdown(f"""
                    <div style="background-color: #f0f9f9; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        <h3 style="color: #1c3e50;">{title}</h3>
                        <p style="color: #24495e;">{abstract_summary}</p>
                    </div>
                """, unsafe_allow_html=True)


# Function to update feedback in session state
def update_feedback(idx):
    st.session_state['feedback'][idx - 1] = st.session_state[f"feedback_{idx}"]
