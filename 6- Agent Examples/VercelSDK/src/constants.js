// Constants for RAG system
export const SIMILARITY_MATCH_COUNT = 5; // Number of similar documents to retrieve
export const EMBEDDING_MODEL_NAME = 'text-embedding-3-small';
export const ANSWERING_MODEL = 'gpt-4o'; // Model used for answering queries
export const SIMILARITY_THRESHOLD = 0.0; // Can increase this to make less permissive 

export const MATCH_THRESHOLD = 0.5;
export const CHUNK_SIZE = 2000; // Approximate chunk size in characters
export const CHUNK_OVERLAP = 100; //Extent of overlap between chujnks

