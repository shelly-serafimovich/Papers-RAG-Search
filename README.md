# README: Retrieval-Augmented Generation (RAG) Pipeline

## Overview
Welcome to our unique implementation of the Retrieval-Augmented Generation (RAG) pipeline! This repository showcases a custom RAG solution built for efficiently querying a large corpus of documents, specifically focusing on the ArXiv dataset. Our pipeline combines state-of-the-art retrieval techniques with generative capabilities to deliver precise and contextually relevant answers.

## Key Features
- **Category-Aware Retrieval**: Our pipeline incorporates a unique category-aware retrieval mechanism. We utilize zero-shot classification to identify the most relevant categories based on the user query and filter results accordingly, which leads to more relevant and focused document retrieval.
- **Adaptive Embedding Refinement**: Using user feedback, we refine the embedding representations to improve accuracy in capturing the user's intent. Our interface allows users to mark retrieved documents as relevant or not, enabling a feedback loop for future optimization.
- **Real-Time ArXiv Integration**: The dataset used in this pipeline is the ArXiv dataset, which updates every few days. The indexing mechanism is designed to ensure the platform stays up-to-date, integrating the most recent articles seamlessly.
- **User-Friendly Interface**: A user interface built using Streamlit makes the interaction straightforward. Users can enter a topic query and receive the 5 most relevant articles along with the abstracts summarized by our LLM.

## Pipeline Architecture
### 1. **Data Collection & Preprocessing**
   The dataset is sourced from ArXiv and filtered based on categories relevant to data science. Preprocessing includes text cleaning, metadata extraction, and embedding generation.

### 2. **Indexing & Retrieval**
   - **Indexing**: Embeddings are generated using the `SentenceTransformer("all-MiniLM-L6-v2")` model. These embeddings are stored in Pinecone, allowing for efficient vector-based similarity searches.
   - **Retrieval**: Upon receiving a query, the system performs two retrieval steps:
     1. **Category Filtering**: Zero-shot classification is used to determine the closest category for the query.
     2. **Similarity Search**: The filtered subset of embeddings is then searched for high similarity using Pinecone.

### 3. **Query Optimization**
   A unique aspect of this pipeline is the use of user feedback to further refine the embeddings. When users interact with the UI and mark articles as relevant or not, the optimized embeddings are saved to ensure better results in future searches.

### 4. **Ranking & Summarization**
   - Articles retrieved are ranked based on their relevance to the query.
   - Summarization of the most relevant abstracts is performed using an LLM to present concise information to the user.

### 5. **User Feedback Loop**
   Our system allows users to provide feedback on the retrieved articles. This feedback is crucial for optimizing future queries, ensuring that the model learns from user interactions to provide better results.

## How to Run the Pipeline
### Prerequisites
- Python 3.8+
- Streamlit
- Pinecone
- Hugging Face Transformers
- Other dependencies listed in `requirements.txt`

### Installation Steps
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Pinecone Index**
   Ensure you have a Pinecone account and create an index. Update the credentials in the `.env` file.

4. **Run Streamlit Interface**
   ```bash
   streamlit run app.py
   ```

## Usage
- Enter a query in the Streamlit interface.
- Review the 5 most relevant articles and their summaries.
- Mark the articles as relevant or not, allowing the system to learn from your preferences.

## Future Improvements
- **Dynamic Category Expansion**: Adding dynamic identification of new categories as ArXiv updates, further improving the retrieval precision.
- **Enhanced User Feedback Integration**: Implementing advanced optimization techniques for embedding based on the user feedback loop to improve personalization.

## Contributing
We welcome contributions! Feel free to open issues or submit pull requests if you have any improvements or suggestions.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

## Acknowledgments
- Special thanks to the Pinecone and Hugging Face teams for their amazing tools.
- Inspired by state-of-the-art advancements in RAG and conversational AI.

