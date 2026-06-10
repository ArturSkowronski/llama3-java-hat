# Error handling standard (llama3-java-hat)

- Never swallow `InterruptedException`: re-interrupt the thread (`Thread.currentThread().interrupt()`) and abort the unit of work.
- Catch the narrowest exception type possible; broad `catch (Exception e)` requires a written justification comment.
- Returning empty/default values on failure is forbidden outside test fixtures — propagate a domain exception.
