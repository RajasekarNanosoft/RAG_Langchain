# RAG_Langchain
RAG using Langchain Chains
Generate descriptions for Database Columns using Naive RAG.
1. Get input metadata(csv/db).
2. Get context or Knowledge base (.pdf, .txt etc).
3. Embed context in vector database.
4. Retrieve relevant chunk of documents using Similarity search from vector database.
5. Augment the retrived context in prompt
6. Retrieve relevant columns for which description has been generated, from vector db.
7. Prompt LLM for generation of descriptions
8. Store it in vector db
