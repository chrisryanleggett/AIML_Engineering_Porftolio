# Python3 Parallel Programming Library Examples

This directory contains examples demonstrating Python's standard libraries for parallel and concurrent programming.

## Standard Libraries

Three main standard libraries for concurrent execution:
- **`_thread`/`threading`**: Low-level and high-level thread interfaces
- **`asyncio`**: Asynchronous execution for non-blocking code
- **`multiprocessing`**: Beyond-thread concurrency using separate processes

## Threading Basics

Examples covering:
- `start_new_thread()` for spawning threads with function and arguments
- Critical section management with `lock.acquire()` and `lock.release()`
- Synchronization primitives: barriers, semaphores, and events

## Asyncio

Examples demonstrating:
- `async` keyword for asynchronous functions
- `asyncio.run()` to execute async methods
- `await` for waiting on async operations
- `create_task()` and `gather()` for parallel task execution

## Multiprocessing

Examples showing:
- Three process creation methods: `spawn` (new interpreter), `fork` (copy current process), `forkserver` (server-based forking)
- Executing code in separate processes or on remote machines

## Process Communication

Examples covering:
- `Queue` for sequential access between processes
- `Pipe` for two-way communication
- Passing communication objects as arguments to coordinate processes

## Worker Pools

Examples demonstrating:
- Creating pools with `Pool(n)`
- Using functional methods like `map()` to distribute work across spawned processes



