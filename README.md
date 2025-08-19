# RAGChatbot

RAGChatbot is a Retrieval-Augmented Generation (RAG) based chatbot designed to answer questions using your own documents and datasets. It leverages Large Language Models (LLMs) and document retrieval techniques to provide accurate, context-aware responses.

## Project Structure

```
RAGChatbot/
├── src/                        # Source code for the chatbot
│   └── (your .py files)
├── tests/                      # Unit and integration tests
│   └── (your test files)
├── data/
│   └── large_hr_policies_dataset/
│       ├── Anti_Harassment_Policy.txt
│       ├── Business_Ethics_Policy.txt
│       ├── Compensation_and_Benefits_Policy.txt
│       ├── Employee_Separation_Policy.txt
│       ├── Equal_Opportunity_Policy.txt
│       ├── Grievance_Redressal_Policy.txt
│       ├── Overtime_Policy.txt
│       ├── Training_and_Development_Policy.txt
│       ├── Whistleblower_Policy.txt
│       └── Work_From_Home_Allowance_Policy.txt
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Features

- **Retrieval-Augmented Generation:** Combines document retrieval with LLMs for accurate answers.
- **Custom Dataset Support:** Easily add your own documents for domain-specific Q&A.
- **Modular Design:** Organized codebase for easy extension and maintenance.
- **Test Coverage:** Includes a structure for unit and integration tests.

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/)

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/pavanchanduri/RAGChatbot.git
    cd RAGChatbot
    ```

2. **Set up a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Add your documents:**
    - Place your text files in the `data/large_hr_policies_dataset/` directory or create your own dataset folder under `data/`.

### Usage

1. **Run the chatbot:**
    ```sh
    1. The chatbot is integrated with AWS Lambda
    2. The preprocessing Lambda runs when there is new data added in S3 and the embeddings are created and added to pinecone DB.
    3. The chatbot html uses API Gateway to access the Lambda handler function that handles the chatbot interactions
    ```

2. **Interact with the chatbot:**
    - Ask questions related to the documents in your dataset.

### Testing

Run all tests using:
```sh
pytest tests/
```

## Customization

- **Add new datasets:** Place additional `.txt` files in the `data/` directory.
- **Modify retrieval or LLM logic:** Edit the relevant modules in `src/`.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)

## Acknowledgements

- Inspired by recent advances in Retrieval-Augmented Generation and LLMs.
- Uses AWS services like Lambda, AWS Bedrock