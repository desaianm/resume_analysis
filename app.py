from dotenv import load_dotenv
import json
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI
from apikey import key2
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO

load_dotenv()


with open('profile_template.json', 'r') as file:
    resume_temp = json.load(file)

with open('posting_template.json', 'r') as file:
    posting_temp = json.load(file)

def retreive_posting(posting):
   return True


#if postings are more than one 
def text_split(resume):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(resume)
    db = FAISS.from_documents(resume,embeddings)


# Convert Resume into Json for saving into database
def convert_resume_json(text):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",temperature=0.1)

    template = """
    You act like advanced Large language model extract info from Text file for Json Template for better responses. use double quotes for property names.
    you will reply with json file only. All other information than property names should go in additional information property. Template is {resume_temp}.  Text to extract information is given below :
    {text}

    """

    prompt = PromptTemplate(template=template, input_variables=["text","resume_temp"])

    chain = LLMChain(llm=llm,prompt=prompt, verbose=True)
    response = chain.run(text=text,resume_temp=resume_temp)
    
    return response

# Convert Posting into Json for saving into database
def convert_posting_json(text):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",temperature=0.2)

    template = """
    You act like advanced LLM Model extract info from Txt file for Json Template for better responses.use double quotes for property names.
    you will reply with json file only. Template is {posting_temp} please don't add anything extra and if irrelevant data which does not belong to any field, don't add it .
    text is given below for that
    to convert: 
    {text}


    """

    prompt = PromptTemplate(template=template, input_variables=["text","posting_temp"])

    chain = LLMChain(llm=llm,prompt=prompt, verbose=True)
    response = chain.run(text=text,posting_temp=posting_temp)
    return response

# matching resume and posting for a Score 
def match_resume(posting,resume):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",temperature=0.6)

    template = """
    You are an advanced ATS  who scores resume comptability with job posting. You have to match  {resume} profile  with job posting {posting} and 
    give output score how much you are likely to select that resume for a JOB.
    follow below rules:
    1 Don't hesitate to assign low scores (such as 10 or 20) if the match between the resume and job posting is minimal.
    2 Conduct a comprehensive analysis of both the resume and the job posting to ensure a thorough evaluation.
    3 Remember that hiring an unsuitable candidate could result in significant financial losses for the company.
    4 Be mindful that the hiring process is extensive and critical, warranting careful selection rather than arbitrary choices.
    5 Provide a final percentage that encapsulates the candidate's compatibility with the job requirements, based  on your detailed match analysis.
    6 Don't give high scores ( such as above 70) unless relevant experience and skills found
    7 Final Answer should be just a Number
    """

    prompt = PromptTemplate(template=template, input_variables=["posting","resume"])

    chain = LLMChain(llm=llm,prompt=prompt, verbose=True)
    response = chain.run(posting=posting,resume=resume)
    return response

# saving resume into database
def save_resume(resume):

    print(json.loads(resume))
    df_r = pd.DataFrame([resume])
    df_r.to_csv('resumes.csv', mode='a', index=True, header=False)
    
# saving job postings into database
def save_posting(posting):

    print(json.loads(posting))
    df_p = pd.DataFrame([posting])
    df_p.to_csv('postings.csv', mode='a', index=True, header=False)

#extract Text from PDF
def extract_text_from_pdf(resume):
    if (resume != None):
        pdf_reader = PdfReader(resume)
        text=""
        for page_num in range(len(pdf_reader.pages)):
                
                page = pdf_reader.pages[page_num]
                
                # Extract text from the page
                text += page.extract_text()
    return text


def gpt4_json_convert(text,temp):
    
    client = OpenAI()

    prompt = f" don't give any explantion. please analyze and convert file data from this {text} into  json template : {temp}  and in response please return only json file please don't enter data in fields if irrelevant to template"

    response = client.chat.completions.create(
            model="gpt-4-0125-preview",
              response_format={ "type": "json_object" },  # Adjust the model identifier as needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ],temperature=0.2
    )
        
    return response.choices[0].message.content

def generate_resume(posting,resume):

    llm = ChatOpenAI(model_name="gpt-4-0125-preview",temperature=0.5)

    template = """
    You are an advanced Resume Builder Tool designed to optimize resumes for specific job postings. Given a resume and a job posting, your task is to tailor the resume to better match the job requirements. Please adhere to the following guidelines:

    Input Resume: {resume}
    Target Job Posting: {posting}
    
    Tailoring Instructions:

    Direct Output: Provide the revised resume directly without any explanation or notes or score.
    Preserve Original Content: Do not introduce elements that are not originally present in the input resume. Enhancements should only utilize existing information.
    Required Modifications: Implement necessary alterations based on the analysis to better align the resume with the job posting.
    Experience and Education: Maintain the original scope of experiences and educational qualifications. Do not fabricate new entries.
    No New Attributes: Avoid adding any new categories or sections not already included in the resume.
    Make new changes to increase analysis score upto 30 percent.
    give output in text format 

    
    """

    prompt = PromptTemplate(template=template, input_variables=["posting","resume","analysis"])

    chain = LLMChain(llm=llm,prompt=prompt, verbose=True)
    response = chain.run(posting=posting,resume=resume)
    return response
    
    
def create_pdf_resume(json_data, output_filename):
    
    pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
    # Set up the document with the specified output filename and page size
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    json_data = json.loads(json_data)

    

    # Define custom styles for the resume
    title_style = styles['Title']
    title_style.alignment = TA_CENTER
    
    heading_style = styles['Heading2']
    heading_style.alignment = TA_CENTER
    
    
    normal_style = styles['Normal']
    normal_style.alignment = TA_LEFT
    
    title_style.fontName = 'Helvetica-Bold'
    heading_style.fontName = 'Helvetica-Bold'
    normal_style.fontName = 'Helvetica'

    # Create a list to hold the elements of the resume
    elements = []
    
    # Add the name and contact details to the resume
    elements.append(Paragraph(json_data['Name'], title_style))
    elements.append(Spacer(1, 12))
    
    if json_data['Summary']:
        elements.append(Paragraph('Summary', heading_style))
        elements.append(Paragraph(json_data['Summary'], normal_style))
        elements.append(Spacer(1, 12))

    # Add contact links
    link_style = ParagraphStyle('link', parent=styles['Heading3'], spaceAfter=5)

    if json_data['links']:
        # Iterate through each link
        for link in json_data['links']:
            for link_name,link_url in link.items():
                # Format the link as "LinkName: URL"
                link_text = f"{link_name}:{link_url}"
                # Add the formatted link to the story
                elements.append(Paragraph(link_text, normal_style))
            elements.append(Spacer(1, 10))
        
    if json_data["Education"]:
    # Add a section heading and content for Education
        elements.append(Paragraph('Education', heading_style))
        for education in json_data['Education']:
            elements.append(Paragraph(f"{education['Degree']} in {education['FieldOfStudy']}, {education['Institution']}", normal_style))
            elements.append(Spacer(1, 6))
            for ach in education['Achievements']:
                elements.append(Paragraph(ach, normal_style))
            elements.append(Spacer(1, 12))
        
    category_style = ParagraphStyle('category', parent=styles['Heading3'])
    
    if json_data['Skills']:
        elements.append(Paragraph('Skills', heading_style))
    # Add a section heading and content for Skills
    for category in json_data['Skills']:
        for category_name, skill_list in category.items():
            # Concatenate all skills in the category into a single string
            skills_text = ', '.join(skill_list)
            
            # Add category name and its skills to the story
            elements.append(Paragraph(category_name + ":" +skills_text, normal_style))
            
    elements.append(Spacer(1, 10))    

    
    if json_data['Experience']:
        elements.append(Paragraph('Professional Experience', heading_style))
        # Add a section heading and content for Experience
        for exp in json_data['Experience']:
            elements.append(Paragraph(f"{exp['Role']} at {exp['CompanyName']}", normal_style))
            elements.append(Paragraph(f"{exp['StartDate']} - {exp['EndDate']}, {exp['Location']}", normal_style))
            for responsibility in exp['Responsibilities']:
                elements.append(Paragraph(responsibility, normal_style))
            elements.append(Spacer(1, 12))
        
    if json_data['Projects']:
        elements.append(Paragraph('Projects', heading_style))
        for exp in json_data['Projects']:
            elements.append(Paragraph(f"{exp['ProjectName']}", normal_style))
            elements.append(Paragraph(f"{exp['StartDate']} - {exp['EndDate']}", normal_style))
            for responsibility in exp['Responsibilities']:
                elements.append(Paragraph(responsibility, normal_style))
            elements.append(Spacer(1, 12))
        
    
    # Add any additional information
    if json_data['AdditionalInformation']:
        elements.append(Paragraph('Additional Information', heading_style))
        elements.append(Paragraph(json_data['AdditionalInformation'], normal_style))
        elements.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(elements)
    
    


def main():

    # Streamlit Interface to upload Resume and Paste Job Posting
    st.set_page_config(page_title="GPT Resume Checker ")
    st.header("GPT Resume Checker 📈")


    # upload file
    posting = st.text_area("Paste your Job Posting here")
    resume = st.file_uploader("Upload your PDF", type="pdf")
    
    if(resume!=None):
        resume = extract_text_from_pdf(resume)
        
        posting1 = gpt4_json_convert(posting,posting_temp)
        resume1 = gpt4_json_convert(resume,resume_temp)
        new_resum = generate_resume(posting1,resume1)
        resu = gpt4_json_convert(new_resum,resume_temp)
        create_pdf_resume(resu,"formatted_resume.pdf")

        with open("formatted_resume.pdf", "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            st.download_button(
                label="Download Resume as PDF",
                data=pdf_bytes,
                file_name="New_resume_GPT.pdf",
                mime="application/pdf"
            )

        #score_details = match_resume(posting1,resume1)
        #st.metric(label="Match Percentage",value=score_details)
        
        
    st.stop()    
    
    
if __name__ == '__main__':
    main()
