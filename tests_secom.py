
from SeCom.secom import SeCom

memory_manager = SeCom(granularity="segment")

conversation_history = [
    [
        "First session of a very looooooong conversation history",
        "The second user-bot turn of the first session",
    ],
    ["Second Session ..."],
]
requests = ["A question regarding the conversation history", "Another question"]
result = memory_manager.get_memory(
    requests, conversation_history, compress_rate=0.9, retrieve_topk=1
)
print(result["retrieved_texts"])