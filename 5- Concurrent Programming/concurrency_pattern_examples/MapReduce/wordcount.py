"""
Map-Reduce processes large datasets by partitioning data into independent chunks,
applying a function in parallel across workers (MAP phase),
then merging partial results into a final output (REDUCE phase).

This example: splits Wikipedia article text, counts words in parallel, combines counts.
Benchmarks with 1, 2, 4, and 8 workers.
"""

import warnings
warnings.filterwarnings('ignore')

import requests
import time
from multiprocessing import Pool, cpu_count
from collections import Counter
import re
import platform

"""
Fetch Wikipedia article text and return as single string
"""
def fetch_wikipedia_article(title="Sinking of the Titanic"):
    response = requests.get("https://en.wikipedia.org/w/api.php", 
        params={'action': 'query', 'format': 'json', 'titles': title, 
                'prop': 'extracts', 'explaintext': True},
        headers={'User-Agent': 'MapReduceProject/1.0'})
    pages = response.json()['query']['pages']
    return pages[list(pages.keys())[0]]['extract']

"""
Split text into equal chunks for parallel processing
"""
def split_text(text, num_chunks):
    chunk_size = len(text) // num_chunks
    return [text[i*chunk_size:(i+1)*chunk_size if i < num_chunks-1 else len(text)] 
            for i in range(num_chunks)]

"""
MAP: Count words in a text chunk (runs in parallel on each worker)
"""
def map_function(text_chunk):
    words = re.findall(r'\b[a-z]+\b', text_chunk.lower())
    return Counter(words)

"""
REDUCE: Combine word counts from all workers into final totals
"""
def reduce_function(counters):
    total = Counter()
    for counter in counters:
        total.update(counter)
    return total

"""
Execute MapReduce: split data, map in parallel, reduce results
"""
def mapreduce_wordcount(text, num_workers):
    chunks = split_text(text, num_workers)
    with Pool(processes=num_workers) as pool:
        mapped = pool.map(map_function, chunks)
    return reduce_function(mapped)

"""
Benchmark MapReduce with different worker counts
"""
if __name__ == "__main__":
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Cores: {cpu_count()}")
    print()
    
    text = fetch_wikipedia_article()
    print(f"Processing {len(text.split())} words")
    
    for workers in [1, 2, 4, 8]:
        start = time.time()
        counts = mapreduce_wordcount(text, workers)
        print(f"{workers} workers: {time.time() - start:.4f}s")
    
    total_words = sum(counts.values())
    print(f"\nTotal word count: {total_words}")
    print(f"Unique words: {len(counts)}")
    print(f"Top 10 words: {counts.most_common(10)}")