/*
  getRagPrompt() builds the prompt sent to the LLM.
    contextString: text from retrieved documents that might answer the question
    question: the user's original question
  Returns a formatted prompt that instructs the LLM to only use the provided context
*/

export function getRagPrompt(contextString, question) {
    return `You are a helpful assistant. Answer the user's question based ONLY on the provided context. If the context doesn't contain the answer, state politely "I'm sorry, I don't have specific information about that in the knowledge base.". Do not make up answers.

Context:
---
${contextString}
---

Question: ${question}
Answer:`;
}

/*
  Helper function for text splitting. Splitting long texts into manageable chunks helps 
  prevent token limit issues with LLMs,  preserves context, and boosts retrieval performance
  for downstream tasks like RAG. simpleTextSplitter() enables splitting text into chunks based 
  on size and overlap,  which improves efficiency and maintains relevant context across text 
  boundaries.

  @param {string} text The input text.
  @param {number} chunkSize Max characters per chunk.
  @param {number} chunkOverlap Characters overlap between chunks.
  @returns {string[]} Array of text chunks.
*/

export function simpleTextSplitter(text, chunkSize, chunkOverlap) {
  const chunks = []; // Array to hold text chunks
  let i = 0;
  while (i < text.length) {
    const end = Math.min(i + chunkSize, text.length);
    chunks.push(text.slice(i, end));
    i += chunkSize - chunkOverlap;
    if (i <= 0) i = end;
  }

  return chunks.filter((chunk) => chunk.trim().length > 0);
}

/*
  combineDocuments() takes an array of document objects (retrieved context docs)
  and joins their 'content' fields into a single string, separated by delimiters.
  This step is crucial for constructing a unified context passage to send to the LLM,
  ensuring the model sees all relevant information at once when generating an answer.
*/
export function combineDocuments(docs) {
    return docs.map((doc) => doc.content).join('\n\n--\n\n');
}