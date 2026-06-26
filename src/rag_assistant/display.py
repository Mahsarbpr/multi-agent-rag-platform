def print_answer_and_sources(result: dict) -> None:
    print("\n--- Answer ---")
    print(result["answer"])
    if "evaluation" in result:
        print("\n--- Evaluation ---")
        print(result["evaluation"])
    print("\n--- Sources ---")
    for source in result["sources"]:
        print(source)