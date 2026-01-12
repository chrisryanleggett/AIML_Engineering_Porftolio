import 'dotenv/config';
import { openai } from '@ai-sdk/openai';
import { embed } from 'ai';

// Public async function to create embeddings from input text using AI SDK
export async function createEmbedding(content) {
    const { embedding } = await embed({
        model: openai.textEmbeddingModel('text-embedding-3-small'),
        value: content,
    });
    return embedding;
}
