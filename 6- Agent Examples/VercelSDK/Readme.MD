This portfolio example illustrates use of the Vercel SDK to abstract logic and simplify agentic functionality. This demonstrates a team of agents with common chat or message bus setup. 

We will start from scratch and create a new repo so you don't need to set anything up. The first agent will be a working customer support agent that can perform real-world use cases such as web search together with a user interface.

We will integrate Retrieval Augmented Generation (RAG) to increase answer accuracy.

Key features of the agent:
- Ability to perform web search
- Ability to make function calls to call external tools
= autonomous agents.  build a team of agents with common chat or message bus setup.

Architecture:
- Basic Express server for API routes
- Basic function calls to enable LLM to call external tools
- Use Scrimba/basic retrieval augmented generation to increase accuracy of responses

[Vercel API Key Features](https://vercel.com/docs/api):
- **Unified Provider Abstraction:** Switch seamlessly among 30+ top AI model providers (OpenAI, Anthropic, Google, etc.) with minimal code changes—eliminates vendor lock-in.
- **Production-Ready Streaming & Tool Calling:** Out-of-the-box support for real-time chat streaming, advanced tool calling with multi-step workflows, and type-safe schemas.
- **Agent & Workflow Support:** Ships with agent abstractions (including ToolLoopAgent), multi-step tool execution, built-in stop conditions, and human-in-the-loop capabilities.
- **Rich UI Integrations:** Framework-agnostic React/Vue/Svelte/Solid hooks make building responsive, streaming chat UIs fast and easy.
- **Structured Output & DevTools:** Generate and stream type-safe structured outputs; debug complex agent flows with dedicated developer tools.






