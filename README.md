# aichamp-assessment-


Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/abhi9716/aichamp-assessment-.git
   ```

2. Navigate to the project directory:
   ```
   cd aichamp-assessment-
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py

   or

   !ANTHROPIC_API_KEY=your_api_key_here uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.
