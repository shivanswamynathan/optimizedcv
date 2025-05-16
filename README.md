# Resume Enhancer

A Django-based API service for enhancing, analyzing, and optimizing resumes using AI. This application provides various endpoints to process resumes, extract structured data from PDFs, enhance resume content, and analyze resume effectiveness against job descriptions.

## Features

- **Resume Extraction**: Convert PDF resumes to structured JSON format
- **Resume Enhancement**: Improve resume content with AI-powered suggestions
- **Resume Analysis**: Evaluate resume against job descriptions
- **Sentence Enhancement**: Optimize individual sentences with different styles
- **MongoDB Integration**: Store and retrieve enhanced resumes

## Tech Stack

- **Backend**: Django, Django REST Framework
- **AI Integration**: LangChain with multiple model support (OpenAI, Gemini, DeepSeek)
- **Data Storage**: MongoDB
- **PDF Processing**: PyPDF2
- **Async Support**: Django ASGI with asyncio

## Project Structure

```
resume_enhancer/
├── manage.py                      # Django management script
├── resume_enhancer/               # Project settings
│   ├── __init__.py
│   ├── asgi.py                   # ASGI configuration
│   ├── settings.py               # Django settings
│   ├── urls.py                   # Main URL routing
│   └── wsgi.py                   # WSGI configuration
├── resume_processor/             # Main application
│   ├── __init__.py
│   ├── apps.py                   # App configuration
│   ├── models.py                 # Database models
│   ├── urls.py                   # API endpoint routes
│   ├── views.py                  # API view handlers
│   └── utils/                    # Utility modules
│       ├── __init__.py
│       ├── analysis.py           # Resume analysis functionality
│       ├── enhance.py            # Resume enhancement functionality
│       ├── extract.py            # PDF extraction functionality
│       ├── llm_logger.py         # LLM interaction logging
│       ├── modelmanager.py       # AI model management
│       ├── mongodb.py            # MongoDB integration
│       └── sentence_enhance.py   # Single sentence enhancement
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- MongoDB
- API Keys for supported AI models (OpenAI, Gemini, DeepSeek)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/resume-enhancer.git
   cd resume-enhancer
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   # Django Settings
   DJANGO_SECRET_KEY=your-secret-key
   
   # MongoDB Settings
   MONGODB_URL_DEVELOPMENT=mongodb://localhost:27017/optimized_cv_dev
   MONGODB_URL_PRODUCTION=mongodb://user:password@hostname:port/optimized_cv_prod
   MONGODB_URL_TEST=mongodb://localhost:27017/optimized_cv_test
   
   # AI Model API Keys
   OPENAI_API_KEY=your-openai-api-key
   GEMINI_API_KEY=your-gemini-api-key
   DEEPSEEK_API_KEY=your-deepseek-api-key
   
   # Model Selection (options: openai, gemini, deepseek)
   MODEL_NAME=openai
   
   # Environment
   NODE_ENV=development
   ```

### Running the Application

1. Ensure MongoDB is running
2. Run migrations:
   ```bash
   python manage.py migrate
   ```

3. Start the development server:
   ```bash
   python manage.py runserver
   ```

The API will be available at `http://127.0.0.1:8000/`.

## API Endpoints

The following endpoints are available for interacting with the Resume Enhancer:

### 1. Extract Resume from PDF

- **Endpoint**: `/resumes/extract/`
- **Method**: POST
- **Purpose**: Extract structured JSON data from a PDF resume
- **Request Format**: Multipart form with 'resume' file field

### 2. Process Resume

- **Endpoint**: `/resumes/process/`
- **Method**: POST
- **Purpose**: Extract and enhance a resume from PDF
- **Request Format**: Multipart form with 'resume' file field and optional 'job_description' field

### 3. Generate AI Resume

- **Endpoint**: `/resumes/generate/`
- **Method**: POST
- **Purpose**: Enhance an existing resume JSON with AI
- **Request Format**: JSON or multipart form with 'resume' JSON and optional 'job_description' field

### 4. Generate Raw Resume

- **Endpoint**: `/resumes/generate/raw/`
- **Method**: POST
- **Purpose**: Process resume JSON without storing in MongoDB
- **Request Format**: JSON and "JD": "none"

### 5. Enhance Sentences

- **Endpoint**: `/resumes/enhance-sentences/`
- **Method**: POST
- **Purpose**: Enhance individual sentences with AI
- **Request Format**: JSON with 'sentences' array, optional 'context' and 'style' fields

### 6. Analyze Resume

- **Endpoint**: `/resumes/analyze/`
- **Method**: POST
- **Purpose**: Analyze resume against a job description
- **Request Format**: JSON or multipart form with 'resume' (file or JSON) and 'job_description' field

### 7. Deep Analysis

- **Endpoint**: `/resumes/analyze/deep/`
- **Method**: POST
- **Purpose**: Get detailed analysis without suggestions
- **Request Format**: JSON or multipart form with 'resume' (file or JSON) and 'job_description' field

### 8. Brief Analysis

- **Endpoint**: `/resumes/analyze/brief/`
- **Method**: POST
- **Purpose**: Get brief analysis with only suggestions
- **Request Format**: JSON or multipart form with 'resume' (file or JSON) and 'job_description' field

## Testing with Postman

### Setting Up Postman

1. Download and install [Postman](https://www.postman.com/downloads/)
2. Create a new collection named "Resume Enhancer API"
3. Set the base URL to `http://127.0.0.1:8000/` (or your deployment URL)

### 1. Testing Resume Extraction Endpoint

1. Create a new request named "Extract Resume from PDF"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/extract/`
4. In the "Body" tab, select "form-data"
5. Add a key "resume" of type "File" and select a PDF resume file
6. Click "Send" to submit the request

Expected response: JSON containing structured resume data extracted from the PDF.

### 2. Testing Resume Processing Endpoint

1. Create a new request named "Process Resume"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/process/`
4. In the "Body" tab, select "form-data"
5. Add a key "resume" of type "File" and select a PDF resume file
6. Optionally add a key "job_description" of type "Text" with a job description
7. Click "Send" to submit the request

Expected response: JSON containing enhanced resume data.

### 3. Testing Generate AI Resume Endpoint

1. Create a new request named "Generate AI Resume"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/generate/`
4. In the "Body" tab, select "raw" and choose "JSON" from the dropdown
5. Paste your resume JSON (you can use the sample provided in `sample_assests/sample_input.json`)
   ```json
   {
     "resume": { /* Your resume JSON here */ },
     "job_description": "Your job description here",
     "user_id": "optional-user-id"
   }
   ```
6. Click "Send" to submit the request

Alternative form-data approach:
1. In the "Body" tab, select "form-data"
2. Add a key "resume" of type "Text" with your JSON resume structure
3. Add a key "job_description" of type "Text" with a job description
4. Optionally add a key "user_id" of type "Text" with a user ID
5. Click "Send" to submit the request

Expected response: JSON containing enhanced resume data, token metrics, and a resume_id from MongoDB.

### 4. Testing Generate Raw Resume Endpoint

1. Create a new request named "Generate Raw Resume"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/generate/raw/`
4. In the "Body" tab, select "raw" and choose "JSON" from the dropdown
5. Paste your resume JSON
   ```json
   {
     "resume": { /* Your resume JSON here */ },
     "job_description": "none"
   }
   ```
6. Click "Send" to submit the request

Expected response: JSON containing processed resume data without storing in MongoDB.

### 5. Testing Enhance Sentences Endpoint

1. Create a new request named "Enhance Sentences"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/enhance-sentences/`
4. In the "Body" tab, select "raw" and choose "JSON" from the dropdown
5. Add the request payload:
   ```json
   {
     "sentences": [
       "Developed a website using React",
       "Managed a team of 5 developers"
     ],
     "context": "Software engineering resume",
     "style": "professional"
   }
   ```
   Available styles: "professional", "concise", "impactful", "technical", "leadership"
6. Click "Send" to submit the request

Expected response: JSON containing the original and enhanced sentences.

### 6. Testing Resume Analysis Endpoint

1. Create a new request named "Analyze Resume"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/analyze/`
4. In the "Body" tab, select "raw" and choose "JSON" from the dropdown
5. Add the request payload:
   ```json
   {
     "resume": { /* Your resume JSON here */ },
     "job_description": "Your job description here"
   }
   ```
6. Click "Send" to submit the request

Alternative approach with PDF:
1. In the "Body" tab, select "form-data"
2. Add a key "resume" of type "File" and select a PDF resume file
3. Add a key "job_description" of type "Text" with a job description
4. Click "Send" to submit the request

Expected response: JSON containing a detailed analysis of the resume against the job description.

### 7. Testing Deep Analysis Endpoint

1. Create a new request named "Deep Analysis"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/analyze/deep/`
4. Configure the body similarly to the Analyze Resume endpoint
5. Click "Send" to submit the request

Expected response: JSON containing a detailed analysis without suggestions.

### 8. Testing Brief Analysis Endpoint

1. Create a new request named "Brief Analysis"
2. Set the method to **POST**
3. Set the URL to `{{base_url}}/resumes/analyze/brief/`
4. Configure the body similarly to the Analyze Resume endpoint
5. Click "Send" to submit the request

Expected response: JSON containing brief analysis with only suggestions.

## Sample Resume JSON Format

The application expects resume data in a specific JSON format. You can use the following sample as a reference:

```json
{
"details":{
  "basics": {
    "name": "John Doe",
    "label": "Software Engineer",
    "email": "john.doe@example.com",
    "phone": "(609) 999-9995",
    "url": "https://johndoe.com",
    "summary": "Experienced software engineer with expertise in Python and React",
    "location": {
      "city": "New York",
      "countryCode": "US"
    },
    "profiles": [
      {
        "network": "linkedin",
        "username": "john.doe",
        "url": "https://linkedin.com/in/john.doe"
      },
      {
        "network": "github",
        "username": "john.doe",
        "url": "https://github.com/john.doe"
      }
    ]
  },
  "work": [
    {
      "name": "Company A",
      "position": "Senior Developer",
      "location": "New York, NY, USA",
      "startDate": "2020-01",
      "endDate": "present",
      "highlights": [
        "Developed and maintained web applications using React and Django",
        "Led a team of 3 developers on a key project"
      ]
    }
  ],
  "education": [
    {
      "institution": "Stanford University",
      "area": "Computer Science",
      "studyType": "BS",
      "startDate": "2012-09",
      "endDate": "2016-06",
      "courses": [
        "Data Structures and Algorithms",
        "Machine Learning"
      ]
    }
  ],
  "skills": [
    {
      "name": "Programming",
      "keywords": [
        "Python",
        "JavaScript",
        "React",
        "Django"
      ]
    }
  ],
  "projects": [
    {
      "name": "Personal Website",
      "description": "Developed a personal portfolio website using React and Tailwind CSS",
      "startDate": "2022-06",
      "endDate": "2022-08"
    }
  ],
"publications": [
        {
            "title": "Deep Learning for Healthcare Applications",
            "publisher": "Springer",
            "publishedDate": "2023-15",
            "publisherUrl": "https://www.springer.com/book/123456",
            "author": "Dr. John Doe",
            "description": [
                "Explores deep learning methods for medical image analysis.",
                "Includes real-world case studies in radiology and pathology."
            ]
        }
    ],
"awards": [
            {
                "title": "Best Data Scientist of the Year",
                "awarder": "Analytics India Magazine"
            }
        ],
    "languages": [
    {
        "name": "English",
        "proficiency": "Fluent"
    },
    {
        "name": "Hindi",
        "proficiency": "Native"
    }
    ],
"certifications": [
    {
        "certificateName": "Google Data Analytics Professional Certificate",
        "completionId": "GDA123456789",
        "date": "2022-01",
        "url": "https://www.coursera.org/account/accomplishments/certificate/GDA123456789"
    },
    {
        "certificateName": "javascript",
        "completionId": "dhf",
        "date": "2023-10",
        "url": "fdsjf"
    }
    ]
},
"Theme":"software_engineer",
    "JD": "Job briefWe are looking for a UI/UX Designer to join our team and help create user-centered designs for our digital products. The UI/UX Designer will be responsible for gathering user requirements, designing graphic elements, and building navigation components. Ultimately, you will design aesthetically pleasing and functional user interfaces that enhance the overall user experience.Responsibilities: Conduct user research and evaluate user feedback Design user interface elements such as menus, tabs, and widgets Develop UI mockups and prototypes that clearly illustrate site functionality Identify and troubleshoot UX issues Collaborate with developers to implement attractive and functional designs Stay up to date with UI trends, best practices, and industry developments Requirements and skills: Proven work experience as a UI/UX Designer or similar role Proficiency in design software such as Figma, Sketch, or Adobe XD Strong portfolio showcasing UI design skills and UX research knowledge Understanding of interaction design principles and user-centered design approach Excellent problem-solving skills and attention to detail Good communication and teamwork skills"
        
    }
```
