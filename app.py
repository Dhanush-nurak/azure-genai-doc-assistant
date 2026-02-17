import os
from rag_pipeline import DocAssistant

def main():
    assistant = DocAssistant()
    
    # Check if we have data
    data_folder = "data"
    files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    
    if not files:
        print("❌ No PDFs found in /data folder. Please add one.")
        return

    # Ingest the first PDF found
    pdf_path = os.path.join(data_folder, files[0])
    assistant.ingest_document(pdf_path)
    
    print("\n🤖 Document Assistant Ready! (Type 'exit' to quit)")
    
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
            
        response = assistant.ask_question(query)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()