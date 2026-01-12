// Application entry point
import { ingestDocuments } from './src/rag/upsertDocuments.js';

console.log('Starting VercelSDK application...');
await ingestDocuments();
console.log('Application completed.');