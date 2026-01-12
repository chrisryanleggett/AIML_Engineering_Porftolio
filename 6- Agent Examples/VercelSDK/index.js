// Application entry point
import 'dotenv/config';
import { ingestDocuments } from './src/rag/upsertDocuments.js';
import { retrieveSimilarDocs } from './src/rag/retrieveSimilarDocs.js';
import {getRagPrompt, combineDocuments} from './src/utils.js';
import { ANSWERING_MODEL } from './src/constants.js';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';

/*
 Build a basic retrieval system
*/

const EMBEDDING_MODEL_NAME = 'text-embedding-3-small';
const aiModel = openai(ANSWERING_MODEL);


const query = "How many houses were damaged during the great fire of london?";

async function main() {
    // Ingest documents into Supabase PostgreSQL database
    await ingestDocuments();
    
    // Retrieve similar documents
    const retrievedDocs = await retrieveSimilarDocs(query);
    
    console.log('\nRetrieved documents:');
    console.log(JSON.stringify(retrievedDocs, null, 2));

    // Combine retrieved documents into a context string
    const contextString = combineDocuments(retrievedDocs);

    //create a prompt including context docs to send to the model
    const prompt = getRagPrompt(contextString, query);

    // send prompt to model to generate response using AI SDK
    const { text } = await generateText({
        model: aiModel,
        prompt: prompt,
    });

    console.log('\nResponse:');
    console.log(text);
    
    console.log('\n' + '='.repeat(60));
    console.log('Application completed.');
    console.log('='.repeat(60));
  
}

main();