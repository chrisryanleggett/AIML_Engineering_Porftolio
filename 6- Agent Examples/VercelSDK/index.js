// Application entry point
import { ingestDocuments } from './src/rag/upsertDocuments.js';
import { retrieveSimilarDocs } from './src/rag/retrieveSimilarDocs.js';

const query = "The customer support agent handles what?";

async function main() {
    // Ingest documents into Supabase
    await ingestDocuments();
    
    // Retrieve similar documents
    const retrievedDocs = await retrieveSimilarDocs(query);
    
    console.log('\nRetrieved documents:');
    console.log(JSON.stringify(retrievedDocs, null, 2));
    
    console.log('\n' + '='.repeat(60));
    console.log('Application completed.');
    console.log('='.repeat(60));
}

main();