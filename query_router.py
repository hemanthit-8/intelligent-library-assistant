def classify_query(query):

    q = query.lower()

    if "how many" in q:
        return "COUNT"

    if "list" in q or "names" in q:
        return "LIST"

    if "domain" in q or "category" in q:
        return "DOMAIN"

    return "RAG"