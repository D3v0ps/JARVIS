"""Language-model side of JARVIS.

reflex        template matcher that answers plain imperatives without the model
``sentences``      streaming sentence splitter plus text cleanup for speech.
``ollama_client``  thin streaming HTTP client for a local Ollama daemon.
``conversation``   rolling history and system-prompt assembly.
``brain``          one voice turn: stream -> sentences -> speech, tools in between.

Nothing in this package touches audio hardware or Windows-only APIs.
"""
