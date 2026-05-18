from openai import OpenAI
import numpy as np
import lmstudio as lms

client = OpenAI(base_url="http://localhost:1234/v1", api_key="test")
EMBED_MODEL = "text-embedding-google_embeddinggemma-300m-qat"

corpus_of_documents = [
    "At a YC event, Brian Chesky gave a memorable talk where he challenged conventional wisdom about running large companies. As Airbnb grew, he received advice to 'hire good people and give them room' which proved disastrous, leading him to develop his own management approach inspired by Steve Jobs.",
    "Many successful founders at the event reported similar experiences - following traditional management advice had damaged their companies instead of helping them grow. This raised questions about why everyone was giving founders the wrong advice.",
    "The answer emerged: founders were being told how to run companies like professional managers, not founders. There are two distinct modes of running a company: founder mode and manager mode. Most assumed scaling meant switching to manager mode.",
    "Founder mode remains largely undocumented - there are no specific books about it, and business schools don't acknowledge its existence. What we know comes from individual founders' experiments and experiences.",
    "The traditional manager approach treats parts of the org chart as black boxes, avoiding 'micromanagement' by delegating completely to direct reports. In practice, this often means hiring professional fakers who can damage the company.",
    "Founders report feeling gaslit both by advisors pushing manager mode and by employees when implementing it. This is a rare case where founders should trust their instincts despite widespread disagreement.",
    "Founder mode breaks the principle that CEOs should only engage via direct reports. Skip-level meetings become normal rather than unusual, opening up many new organizational possibilities.",
    "Steve Jobs' example of running annual retreats for Apple's 100 most important people (not necessarily highest-ranking) demonstrates an unconventional approach that could make big companies feel like startups.",
    "While founders can't run a 2000-person company exactly like a 20-person startup, the extent and nature of delegation in founder mode will vary by company and situation, making it more complex than manager mode.",
    "Early evidence suggests founder mode works better than manager mode, based on examples of founders who've found their way toward it, even when their methods were considered eccentric.",
    "The premise that founders must run their companies as managers has been accepted even in Silicon Valley. The dismay of founders who tried this approach and their success in finding alternatives suggests another way exists.",
    "Business education and literature focus almost exclusively on manager mode, leaving a gap in understanding how founders can effectively run larger companies while maintaining their unique advantages.",
    "Brian Chesky's success at Airbnb, demonstrated by their exceptional free cash flow margin, suggests that founder mode can produce superior results when properly implemented.",
    "The insight about different modes of company operation came from observing patterns in founder experiences, particularly their consistent struggles with conventional management wisdom.",
    "Founder mode may involve more direct engagement across organizational levels, breaking traditional management hierarchies while maintaining necessary delegation structures.",
    "The skills and approaches that work for professional managers may be fundamentally different from what works for founders, suggesting the need for distinct operational frameworks.",
    "The success of founders who rejected traditional management advice indicates that founder mode, while less understood, might be more effective for scaling companies.",
    "The lack of understanding about founder mode represents both a challenge and an opportunity - founders have achieved success despite following suboptimal advice.",
    "The story suggests that many successful founders may have independently discovered aspects of founder mode while being viewed as unconventional or difficult.",
    "The potential for improved company performance once founder mode is better understood and documented could lead to significant changes in how fast-growing companies are managed."
]


embedding_model = lms.embedding_model(EMBED_MODEL)

def embed_document(chunk: str) -> np.ndarray:
    results = embedding_model.embed(f"search_document: {chunk}")
    return np.array(results, dtype=np.float32)

def embed_query(query: str) -> np.ndarray:
    results = embedding_model.embed(f"search_query: {query}")
    return np.array(results, dtype=np.float32)

def search(query, data_source, k=5):
    """
    Search function to find top k similar chunks to the query
    """
    # Compute embedding for the query
    query_embedding = embed_query(query)

    # Normalize the query embedding
    query_norm = np.linalg.norm(query_embedding)

    # Compute cosine similarities between query embedding and data source embeddings
    similarities = []
    for embedding, chunk in data_source:
        # Normalize the data source embedding
        embedding_norm = np.linalg.norm(embedding)

        # Compute cosine similarity
        similarity = np.dot(query_embedding, embedding) / (query_norm * embedding_norm)
        similarities.append((similarity, chunk))

    # Sort the similarities in descending order
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Get the top k chunks
    top_k = similarities[:k]

    return top_k


# Main execution
if __name__ == "__main__":
    # Pre-embed documents
    embedded_data_source = []
    for chunk in corpus_of_documents:
        embedding = embed_document(chunk)
        embedded_data_source.append((embedding, chunk))

    # Search
    user_query = "explain what is founder mode"
    top_k_chunks = search(user_query, embedded_data_source, k=5)

    # Extract chunks
    retrieved_chunks = [chunk for similarity, chunk in top_k_chunks]

    # Build prompt
    base_prompt = """You are an AI assistant for RAG. Your task is to understand the user question, and provide an answer using the provided contexts.

Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, "The provided context does not have the answer."

User question: {user_query}

Contexts:
{chunks_information}
"""

    prompt = base_prompt.format(user_query=user_query, chunks_information="\n".join(retrieved_chunks))

    print("\n\nPROMPT: ", prompt)

    # Get LLM response
    response = client.chat.completions.create(
        model="google/gemma-4-e4b",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    print("\n\nANSWER: ", response.choices[0].message.content)