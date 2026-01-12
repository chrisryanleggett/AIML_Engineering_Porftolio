-- Create the function to search documents using cosine similarity
-- (Drop the function first if you are recreating it)
-- drop function if exists match_documents;

create function match_documents (
    query_embedding vector(1536), --matches the embedding dimensions
    match_count int DEFAULT 5, -- set default results to return to 5
    match_threshold float DEFAULT 0.3, -- Ensures that only documents that have a minimum similarity to the query_embedding are returned. 
    filter jsonb DEFAULT '{}'
)   returns table (
    id bigint,
    content text,
    metadata jsonb, --return metadata along with the content
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        documents.id,
        documents.content,
        documents.metadata,
        --calculate cosine similarity  (1 - cosine distance)
        -- <-> is the cosine distance operator
        1 - (documents.embedding <-> query_embedding) as similarity
    from documents
    -- Optional where clause to filter by metadata if needed in the future
    -- e.g. where documents.metadata @> filter
    where 1 - (documents.embedding <-> query_embedding) > match_threshold
    order by documents.embedding <-> query_embedding -- order by cosine distance (closest first)
    limit match_count;
end;
$$;