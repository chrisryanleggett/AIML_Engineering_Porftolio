import { ingestDocuments } from './upsertDocuments.js';

// Imports for the query and retrieval part of the RAG pipeline
import { retrieveSimilarDocs } from './retrieveSimilarDocs.js';
import { getRagPrompt, combineDocuments } from '../utils.js';
import { ANSWERING_MODEL } from '../constants.js';

const query = "In 1843, what was the key milestone in computing?";

async function main(query) {
    // Retrieve docs that contain content relevant to the query
    const retrievedDocs = await retrieveSimilarDocs(query);
    console.log(retrievedDocs);

    // Create a prompt including context docs to send to the model
    // const contextString = combineDocuments(retrievedDocs);

    // Create a prompt including context docs to send to the model
    // const prompt = getRagPrompt(contextString, query);

    // console.log(`Prompt: ${prompt}`);

    // Send prompt to model to generate response
    // const response = await openai.responses.create({
    //     model: ANSWERING_MODEL,
    //     input: query
    // });

    // console.log(response.output_text);
}

main(query);

// Uncomment to run document ingestion instead
// async function ingest() {
//     await ingestDocuments();
// }
// ingest()
