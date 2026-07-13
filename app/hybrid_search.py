from rank_bm25 import BM25Okapi


def bm25_search(query, chunks, top_k=5):
    """
    Keyword search using BM25
    """

    if not chunks:
        return []

    tokenized_chunks = [
        chunk.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]