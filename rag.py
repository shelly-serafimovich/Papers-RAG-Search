from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import cohere
import google.generativeai as genai
from pinecone import Pinecone
import numpy as np
import os


# Load models and initialize Pinecone index
model = SentenceTransformer("all-MiniLM-L6-v2")
pinecone_api_key = os.getenv('PINECONE')
co_api_key = os.getenv('CO')



# Function to encode text using SentenceTransformer
def encode_text(text):
    embeddings = model.encode(text)
    return embeddings


# Function to call OpenAI GPT API
def call_gpt_api(query, abstract):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a relevance evaluator for scientific articles. Reply with only a "
                                          "single number between 1 and 10."},
            {"role": "user", "content": f"Question: {query}\nArticle Abstract: {abstract}\nRate the relevance from 1 "
                                        f"to 10. Only provide the number as an answer."}
        ]
    )
    score_text = response['choices'][0]['message']['content']
    return float(score_text.strip())


# Function to call Cohere API
def call_cohere_api(query, abstract):
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=f"You are a relevance evaluator for scientific articles. Question: {query}\nArticle Abstract: "
               f"{abstract}\nRate the relevance from 1 to 10. Only provide the number as an answer.",
        max_tokens=10
    )
    return float(response.generations[0].text.strip())


# Function to call Gemini API
def call_gemini_api(query, abstract):
    prompt_text = f"You are a relevance evaluator for scientific articles. Question: {query}\nArticle Abstract: " \
                  f"{abstract}\nRate the relevance from 1 to 10. Only provide the number as an answer."
    response = genai_client.generate_content(prompt_text)
    return float(response.text)


# Function to retrieve articles from Pinecone
def retrieve_articles(query, top_k=5):
    query_embedding = encode_text(query)
    response = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True, include_values=True)
    articles = [(match['metadata']['title'], match['metadata']['abstract'], match['values']) for match in
                response['matches']]
    return articles


# Function to evaluate articles using GPT, Cohere, and Gemini
def evaluate_articles(query, articles):
    gpt_scores, cohere_scores, gemini_scores = [], [], []
    for (title, abstract, values) in articles:
        combined_text = f"Title: {title}\nAbstract: {abstract}"

        # OpenAI GPT Score
        gpt_score = call_gpt_api(query, combined_text)
        gpt_scores.append(gpt_score)

        # Cohere Score
        cohere_score = call_cohere_api(query, combined_text)
        cohere_scores.append(cohere_score)

        # Gemini Score
        gemini_score = call_gemini_api(query, combined_text)
        gemini_scores.append(gemini_score)

    avg_scores = [(g + c + a) / 3 for g, c, a in zip(gpt_scores, cohere_scores, gemini_scores)]
    return avg_scores


# Function to expand the query using GPT
def expand_query(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in data science and information retrieval."},
            {"role": "user",
             "content": f"Provide additional keywords or phrases that could be incorporated into the following query "
                        f"to refine or broaden it, but do not answer the question. Return a concise list of relevant "
                        f"terms or phrases only. Query: '{query}'"}
        ]
    )
    expanded_terms = response['choices'][0]['message']['content'].strip()
    expanded_query = f"{query} {expanded_terms}"
    return expanded_query


# Function to calculate cosine similarity
def calculate_cosine_similarity(query_embedding, abstract_embedding):
    return cosine_similarity([query_embedding], [abstract_embedding])[0][0]


# Function to get the top 5 articles based on model score, combining original and expanded queries
def get_top_5_articles(query, top_k=5):
    # Retrieve articles and scores for the original query
    original_articles = retrieve_articles(query, top_k)
    original_scores = evaluate_articles(query, original_articles)

    # Retrieve articles and scores for the expanded query
    expanded_query = expand_query(query)
    expanded_articles = retrieve_articles(expanded_query, top_k)
    expanded_scores = evaluate_articles(query, expanded_articles)

    # Combine results from original and expanded queries
    combined_articles = []
    for i, (title, abstract, values) in enumerate(original_articles):
        combined_articles.append({
            "title": title,
            "abstract": abstract,
            "embedding": values,
            "model_score": original_scores[i],
            "source_query": "original"
        })

    for i, (title, abstract, values) in enumerate(expanded_articles):
        # Avoid duplicates
        if title not in [article["title"] for article in combined_articles]:
            combined_articles.append({
                "title": title,
                "abstract": abstract,
                "embedding": values,
                "model_score": expanded_scores[i],
                "source_query": "expanded"
            })

    # Sort combined articles by model score and select the top 5
    top_5_articles = sorted(combined_articles, key=lambda x: x["model_score"], reverse=True)[:5]
    return top_5_articles


# Function to refine the query vector based on feedback
def refine_query_with_feedback(article_embeddings, feedback, weight_relevance=1.0, weight_irrelevance=0.1):
    # Separate embeddings based on feedback
    relevant_embeddings = [emb for emb, fb in zip(article_embeddings, feedback) if fb == 1]
    irrelevant_embeddings = [emb for emb, fb in zip(article_embeddings, feedback) if fb == 0]

    # Compute the refined query vector
    if relevant_embeddings:
        relevant_vector = np.mean(relevant_embeddings, axis=0) * weight_relevance
    else:
        # Fallback if no relevant feedback is given
        relevant_vector = np.mean(article_embeddings, axis=0)

    # Optionally adjust based on irrelevant embeddings
    if irrelevant_embeddings:
        irrelevant_vector = np.mean(irrelevant_embeddings, axis=0) * weight_irrelevance
        final_vector = relevant_vector - irrelevant_vector  # Subtract irrelevant components
    else:
        final_vector = relevant_vector

    improved_articles = retrieve_improved_results(final_vector, top_k=5)
    improved_summarized_results = summarize_abstracts_with_gpt(improved_articles)

    return improved_summarized_results


# Function to retrieve new articles using the refined vector
def retrieve_improved_results(final_vector, top_k=5):
    # Query Pinecone using the refined vector
    query_result = index.query(vector=final_vector.tolist(), top_k=top_k, include_metadata=True, include_values=True)

    # Extract metadata from results
    articles = []
    for match in query_result['matches']:
        articles.append({
            "title": match['metadata'].get('title', 'No title available'),
            "abstract": match['metadata'].get('abstract', 'No abstract available'),
            "embedding": match['values'],
            "model_score": 1,
            "source_query": "expanded"
        })

    return articles


# Function to summarize abstracts using OpenAI's GPT-3.5
def summarize_abstracts_with_gpt(top_5_articles):
    for article in top_5_articles:
        abstract = article['abstract']
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes scientific abstracts."},
                {"role": "user", "content": f"Please provide a brief summary of the following abstract:\n\n{abstract}"}
            ]
        )
        # Extract the summary text from the response
        summary = response['choices'][0]['message']['content'].strip()
        article["summary"] = summary
    return top_5_articles


def find_relevant_articles(query):
    top_5 = get_top_5_articles(query)
    # title_abstract_dict = {article['title']: article['abstract'] for article in top_5}
    summarized_results = summarize_abstracts_with_gpt(top_5)
    return summarized_results


if __name__ == "__main__":
    article_res = find_relevant_articles(query="What are the effects of climate change on biodiversity?")
    print("top5")
    improved_articles = refine_query_with_feedback(article_embeddings=[article['embedding'] for article in article_res], feedback=[1, 0, 1, 0, 1])
    print("finished noder neder")

