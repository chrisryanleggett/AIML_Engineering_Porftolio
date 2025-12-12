# Concurrency Pattern Examples

This directory contains C++ and Python code examples demonstrating five common concurrency patterns essential for parallel programming. These examples are included to help new developers understand key concepts in concurrent programming and how to apply them in practice.

For broader context on concurrent programming concepts, see the [main README](../README.MD).

## Common Concurrency Patterns with Examples

**Divide and Conquer**: Splits data into smaller pieces, solves individually, then combines results. Best for sorted data.

**Map-Reduce**: A subset of divide and conquer where data breaks into individual points, same operation applies to all (map), then results combine into single value (reduce). Scales well regardless of data size.

**Repository Pattern**: Maintains shared state across threads/processes. Processes run independently and request/update central repository atomically.

**Pipelines/Workflows**: Directed graphs where data flows through steps. Pipelines are linear; workflows allow fan-out/fan-in patterns with multiple parallel paths.

**Recursion**: Solves problems by calling function on itself with smaller input until base case. Common in functional programming; less suitable for distributed systems or GPUs.