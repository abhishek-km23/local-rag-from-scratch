from rag_pipeline import answer_question

def main():
    print("PDF RAG is ready.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            print("Exiting.")
            break

        try:
            answer = answer_question(question)
            print("\nAnswer:\n", answer)
            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            print("ERROR:", e)


if __name__ == "__main__":
    main()
