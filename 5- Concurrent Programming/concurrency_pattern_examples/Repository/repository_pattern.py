"""
Repository Pattern: Central store maintains data atomically while processes maintain their own state.
Processes may work on data for a while before updating, risking overwriting changes made by others.
Version checking prevents stale updates - write fails if repository changed during process work time.
"""

"""
Demonstrates repository pattern using Wikipedia article word counting across 4 processes.
Processes with different work delays create version conflicts when slow processes
attempt to write stale data, triggering conflict detection and retry logic.
"""

import warnings
warnings.filterwarnings('ignore')
import requests
import time
from multiprocessing import Process, Manager
from collections import Counter
import re

# Repository class with atomic operations and version tracking
class ArticleRepository:
    def __init__(self, manager):
        self.data = manager.dict()
        self.lock = manager.Lock()
        self.version = manager.Value('i', 0)
    
    # Atomically read data and current version
    def read(self):
        with self.lock:
            return dict(self.data), self.version.value
    
    # Atomically write only if version matches (detects staleness)
    def write(self, new_data, expected_version):
        with self.lock:
            if self.version.value != expected_version:
                return False
            self.data.update(new_data)
            self.version.value += 1
            return True

# Fetch Wikipedia article text
def fetch_wikipedia_article(title="Sinking of the Titanic"):
    response = requests.get("https://en.wikipedia.org/w/api.php",
        params={'action': 'query', 'format': 'json', 'titles': title,
                'prop': 'extracts', 'explaintext': True},
        headers={'User-Agent': 'RepositoryPattern/1.0'})
    pages = response.json()['query']['pages']
    return pages[list(pages.keys())[0]]['extract']

# Process reads from repository, maintains own state during work, then writes back
def analyze_section(repo, process_id, start, length, delay):
    # Read data snapshot with version
    data, version = repo.read()
    print(f"Process {process_id}: Read version {version}")
    
    # Process maintains own state - count words in section
    section = data['text'][start:start + length]
    words = re.findall(r'\b[a-z]+\b', section.lower())
    word_count = Counter(words)
    time.sleep(delay)
    
    # Attempt to write back results with version check
    data, _ = repo.read()
    data['word_counts'].update(word_count)
    
    if repo.write(data, version):
        print(f"Process {process_id}: Write succeeded")
    else:
        # Handle conflict - data became stale during work
        print(f"Process {process_id}: CONFLICT - retrying")
        data, _ = repo.read()
        data['word_counts'].update(word_count)
        repo.write(data, repo.read()[1])

if __name__ == "__main__":
    # Initialize repository with article data
    text = fetch_wikipedia_article()
    manager = Manager()
    repo = ArticleRepository(manager)
    repo.write({'text': text, 'word_counts': Counter()}, 0)
    
    print(f"Processing {len(text.split())} words\n")
    
    # Spawn 4 processes with different work delays (creates staleness)
    processes = []
    section_size = len(text) // 4
    for i, delay in enumerate([0.5, 0.1, 0.3, 0.2]):
        p = Process(target=analyze_section, 
                   args=(repo, i+1, i*section_size, section_size, delay))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Read final results from repository
    final_data, final_version = repo.read()
    print(f"\nFinal version: {final_version}")
    print(f"Total words: {sum(final_data['word_counts'].values())}")
    print(f"Top 10: {final_data['word_counts'].most_common(10)}")