# Resume Analyzer

## Project Description

This project is designed to analyze resumes and job postings, converting them into structured JSON formats for easier comparison and scoring. It leverages the power of OpenAI's GPT models to generate tailored resumes based on job postings, ensuring a better match between candidates and job requirements.

## Features

- **Resume and Job Posting Extraction**: Extracts text from PDFs and converts job postings into structured JSON.
- **Text Splitting and Embedding**: Splits large texts into manageable chunks and generates embeddings for analysis.
- **Resume Optimization**: Tailors resumes to better match specific job postings using GPT-3.5.
- **Scoring System**: Matches resumes with job postings and provides a compatibility score.
- **PDF Resume Generation**: Creates a professional PDF resume based on the optimized content.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the web interface.
- **PyPDF2**: For PDF text extraction.
- **Pandas**: For handling data structures.
- **Langchain**: For text splitting and embedding.
- **OpenAI API**: For text generation and analysis.
- **ReportLab**: For generating PDF resumes.
- **dotenv**: For loading environment variables.

## Setup

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Create a `.env` file in the root directory and add the following:
    ```env
    OPENROUTER_API_KEY=your_openrouter_api_key
    OPENAI_API_KEY=your_openai_api_key
    RAPIDAPI_KEY=your_rapidapi_key
    ```

4. **Download the required models**:
    Ensure you have the necessary HuggingFace embeddings and Langchain models downloaded.

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2. **Use the Web Interface**:
    - Upload your PDF resume.
    - Paste your job posting text into the provided area.
    - View the generated analysis and optimized resume.
    - Download the tailored resume as a PDF.

## Example

1. **Uploading Resume and Posting**:
    - Upload your resume PDF file using the file uploader.
    - Paste the job posting text into the text area.

2. **Viewing Analysis and Optimized Resume**:
    - The application will display the analysis score and the newly generated resume.

3. **Downloading the PDF**:
    - Click the download button to save the optimized resume as a PDF.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING](CONTRIBUTING.md) guidelines first.

## Acknowledgements

- [OpenAI](https://openai.com) for providing the GPT-4 API.
- [HuggingFace](https://huggingface.co) for embeddings.
- [Langchain](https://langchain.org) for text processing tools.
- [ReportLab](https://www.reportlab.com) for PDF generation.

## Contact

For any questions or suggestions, please contact adcan288@gmail.com.
