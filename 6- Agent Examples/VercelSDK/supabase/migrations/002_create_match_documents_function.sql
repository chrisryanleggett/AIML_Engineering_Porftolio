-- Create the function to search documents using cosine similarity
-- (Drop the function first if you are recreating it)
-- drop function if exists match_documents;

create function match_documents (
    query
)