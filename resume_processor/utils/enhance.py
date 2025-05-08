import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
import os
from .modelmanager import SimpleModelManager
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
from langchain.output_parsers import PydanticOutputParser
logger = logging.getLogger(__name__)


# Define Pydantic models for resume sections

class EnhancedSummary(BaseModel):
    optimized_summary: str = Field(description="Rewritten summary that follows the STRICT word limit and content constraints.")

class EnhancedExperience(BaseModel):
    optimized_experience: List[str] = Field(description="List of EXACTLY 5 bullet points per experience entry, formatted according to the given constraints.")

class EnhancedEducation(BaseModel):
    optimized_education: List[str] = Field(description="List of EXACTLY 3 education bullet points with STRICT adherence to word limits and abbreviations.")

class EnhancedProjects(BaseModel):
    optimized_projects: List[str] = Field(description="List of EXACTLY 5 formatted project bullet points with technical depth.")

class EnhancedSkills(BaseModel):
    optimized_skills: Dict[str, List[str]] = Field(description="Skills categorized into EXACTLY 4 categories, each containing EXACTLY 4 specialized skills.")

class EnhancedAwards(BaseModel):
    optimized_awards: List[str] = Field(description="List of optimized award descriptions following the STRICT format.")

class EnhancedResume(BaseModel):
    optimized_summary: Optional[str] = None
    optimized_experience: Optional[List[str]] = None
    optimized_education: Optional[List[str]] = None
    optimized_projects: Optional[List[str]] = None
    optimized_skills: Optional[Dict[str, List[str]]] = None
    optimized_awards: Optional[List[str]] = None

    @validator('optimized_experience')
    def validate_experience_format(cls, v):
        if v is not None:
            # Validate each experience entry has exactly 5 bullet points
            if not all(isinstance(entry, str) for entry in v):
                raise ValueError("All experience entries must be strings")
        return v

    @validator('optimized_education')
    def validate_education_format(cls, v):
        if v is not None:
            # Validate education has exactly 3 bullet points
            if not all(isinstance(entry, str) for entry in v):
                raise ValueError("All education entries must be strings")
        return v

    @validator('optimized_skills')
    def validate_skills_format(cls, v):
        if v is not None:
            # Validate skills structure
            for category, skills in v.items():
                if not isinstance(skills, list):
                    raise ValueError(f"Skills for category '{category}' must be a list")
        return v


TEMPLATE_PROMPTS = {
    "simple": {
        "parameters": [
            {
                "name": "summary",
                "type": "string",
                "description": "Generate a concise and impactful summary of STRICTLY 45-60 words. Highlight ONLY core skills, key strengths, and MEASURABLE impact. AVOID fluff or vague language. Use PRECISE, ATS-friendly terminology."
            },
            {
                "name": "experience",
                "type": "string",
                "description": "Each experience section MUST contain EXACTLY 5 bullet points. DEFAULT to two-line entries (STRICTLY 28-32 words each) showcasing applied skills and QUANTIFIABLE results. ONLY use one-line entries (STRICTLY 15-16 words) if expansion is IMPOSSIBLE. EACH bullet MUST begin with a STRONG action verb."
            },
            {
                "name": "education",
                "type": "string",
                "description": "Provide EXACTLY 3 bullet points for education details. EACH point MUST be STRICTLY 10-13 words, emphasizing MAJOR achievements, key skills, or projects. Use ONLY degree abbreviations such as 'BSc', 'MSc', 'BA', 'MA', and 'BE' for UNIFORMITY."
            },
            {
                "name": "projects",
                "type": "string",
                "description": "Each project section MUST include EXACTLY 5 bullet points. DEFAULT to two-line entries (STRICTLY 28-32 words) emphasizing impact, technologies used, and MEASURABLE outcomes. ONLY use one-line entries (STRICTLY 15-16 words) when expansion is IMPOSSIBLE. EACH bullet MUST begin with a STRONG action verb."
            },
            {
                "name": "skills",
                "type": "string",
                "description": "STRICTLY organize skills into EXACTLY 4 categories, listing EXACTLY 4 core skills per category. USE ONLY concise skill names. NO descriptions, NO explanations, and NO extra words ALLOWED."
            },
            {
                "name": "awards",
                "type": "string",
                "description": "Summarize EACH award in a SINGLE, IMPACTFUL sentence of STRICTLY 20-25 words. EMPHASIZE significance, outcome, or recognition criteria. EACH summary MUST include AT LEAST ONE metric or QUANTIFIABLE impact."
            }
        ],
        "prompt_template": "Analyze the following resume details and optimize them according to STRICT formatting constraints:\n\nSummary: {{summary}}\n\nExperience:\n{{experience}}\n\nEducation:\n{{education}}\n\nProjects:\n{{projects}}\n\nSkills:\n{{skills}}\n\nAwards:\n{{awards}}\n\nSTRICTLY enforce all length limits, structural constraints, and ATS optimization techniques. Ensure all sections adhere to the defined rules without exception.",
        "response_schema": {
            "type": "object",
            "properties": {
                "optimized_summary": {
                    "type": "string",
                    "description": "Rewritten summary that follows the STRICT word limit and content constraints."
                },
                "optimized_experience": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of EXACTLY 5 bullet points per experience entry, formatted according to the given constraints."
                },
                "optimized_education": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of EXACTLY 3 education bullet points with STRICT adherence to word limits and abbreviations."
                },
                "optimized_projects": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of EXACTLY 5 formatted project bullet points with technical depth."
                },
                "optimized_skills": {
                    "type": "object",
                    "description": "Skills categorized into EXACTLY 4 categories, each containing EXACTLY 4 specialized skills."
                },
                "optimized_awards": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of optimized award descriptions following the STRICT format."
                }
            }
        }
    },
    "software_engineer": {
    "parameters": [
        {
        "name": "summary",
        "type": "string",
        "description": "Generate a technical summary of exactly 55-60 words focusing on specialized programming skills, achievements, and measurable impact. MUST incorporate at least 5 ATS-relevant technical keywords for improved searchability."
        },
        {
        "name": "experience",
        "type": "string",
        "description": "Each experience section MUST contain exactly 5 bullet points. DEFAULT to two-line entries (28-32 words) that demonstrate technical challenges solved and measurable outcomes. Only use one-line entries (15-16 words) when content absolutely cannot be expanded. Each bullet MUST start with a technical action verb."
        },
        {
        "name": "education",
        "type": "string",
        "description": "Degree name should be in short like ME, BE. Education details MUST include exactly 3 bullet points. Each point MUST be 10-13 words long, highlighting relevant technical coursework, certifications, or achievements. Strictly use degree abbreviations like 'BSc', 'MSc', 'BE', 'BTech', etc."
        },
        {
        "name": "projects",
        "type": "string",
        "description": "Each project section MUST include exactly 5 bullet points. DEFAULT to two-line entries (28-32 words) focusing on technical challenges solved and quantifiable outcomes. Only use one-line entries (15-16 words) when content absolutely cannot be expanded. Each bullet MUST include at least one technical term or technology."
        },
        {
        "name": "skills",
        "type": "string",
        "description": "List skills under exactly 5 technical categories with exactly 4 specialized skills per category. Each skill MUST be a specific technology, language, framework, or methodology relevant to software engineering."
        },
        {
        "name": "awards",
        "type": "string",
        "description": "Summarize each award in a single impactful sentence of exactly 20-25 words highlighting technical achievements, innovation metrics, or leadership outcomes. Each summary MUST include at least one technical term or quantifiable result."
        }
    ],
    "prompt_template": "Analyze the following resume details and optimize them according to ATS-friendly software engineering standards:\n\nSummary: {{summary}}\n\nExperience:\n{{experience}}\n\nEducation:\n{{education}}\n\nProjects:\n{{projects}}\n\nSkills:\n{{skills}}\n\nAwards:\n{{awards}}\n\nEnsure each section follows strict formatting rules, maintains technical depth, and maximizes keyword optimization for better searchability.",
    "response_schema": {
        "type": "object",
        "properties": {
        "optimized_summary": {
            "type": "string",
            "description": "Rewritten summary that follows the required format."
        },
        "optimized_experience": {
            "type": "array",
            "items": {
            "type": "string"
            },
            "description": "List of exactly 5 bullet points for each experience entry, formatted according to the given constraints."
        },
        "optimized_education": {
            "type": "array",
            "items": {
            "type": "string"
            },
            "description": "List of 3 education bullet points with strict adherence to word limits and abbreviations."
        },
        "optimized_projects": {
            "type": "array",
            "items": {
            "type": "string"
            },
            "description": "List of 5 formatted project bullet points with technical depth."
        },
        "optimized_skills": {
            "type": "object",
            "description": "Skills categorized into exactly 5 technical categories, each containing exactly 4 specialized skills."
        },
        "optimized_awards": {
            "type": "array",
            "items": {
            "type": "string"
            },
            "description": "List of optimized award descriptions following the required format."
        }
        }
    }
    }

}

SECTION_PROMPTS = {
    "basics": """
        Enhance the resume summary section by making it concise, professional, and results-driven.  
        If a job description is provided, tailor the summary to align with relevant skills and requirements.  
        If a job description is NOT provided, improve clarity and impact while keeping the content natural.
        
        INSTRUCTIONS:
        **STRICT RULES TO FOLLOW:** 
        - **Start with a strong opening sentence** highlighting years of experience and core expertise.
        - **Emphasize key skills, technologies, and industry knowledge.** 
        - CRITICAL: {summary_description}
        - **Include at most ONE measurable outcome (if applicable) but avoid forcing metrics.**
        - **Align with the job description if provided.** 
        - **Keep all other personal information (name, contact, location) unchanged.**

    **FORMAT EXAMPLES (Before & After):**  

      **Before:**  
      "Experienced AI engineer with knowledge of Python and ML. Skilled in automation and data processing. Looking for opportunities in AI and software development."  

      **After:**  
      "**AI Engineer** with **3+ years of experience** specializing in **Machine Learning, NLP, and AI-driven automation**. Proficient in **Python, TensorFlow, and LangChain**, with expertise in **developing scalable AI solutions**. Successfully **optimized LLM models**, improving **inference speed by 30%**, enhancing real-time application performance. Passionate about **building AI-driven applications** to solve complex business challenges."
        
        JOB DESCRIPTION CONTEXT:
        {job_description}
        
        ORIGINAL CONTENT:
        {original_content}
        
        **Ensure a professional and concise summary with at most one measurable outcome.**  
        **Return ONLY the enhanced JSON for this section.** 
        **Maintain the EXACT SAME structure but enhance the content.**
    """,
    
    "work": """
        Enhance the resume experience section by making it more impactful and results-driven.  
        If a job description is provided, tailor the experience to align with key skills and requirements.  
        If a job description is NOT provided, improve the content while keeping it relevant and concise. 

        INSTRUCTIONS:
        **STRICT RULES TO FOLLOW:** 
        - CRITICAL: {experience_description}
        - **Treat each job experience separately** and enhance it individually.  
        - **Each bullet must be a concise achievement statement (28-32 words max).**  
        - **Start each bullet point with a strong action verb**. 
        - **DEFAULT to two-line entries (28-32 words) unless content absolutely cannot be expanded.**
        - **Include measurable outcomes in exactly TWO bullet points per job—no more.** 
        - **Ensure the remaining bullet points focus on skills, tools, and contributions WITHOUT measurable outcomes.**  
        - **Do not force quantification on every point. Some points should highlight impact in qualitative terms.** 
        - **Avoid vague descriptions; focus on impact and results.**
        - **Keep job titles, company names, and dates unchanged.** 


    **FORMAT EXAMPLES (Before & After):**  
      **Before:**  
      - Managed project workflows.  
      - Developed software tools.  
      - Led a team of engineers.  
      - Improved system performance.  
      - Wrote documentation.  

      **After:**  
      - Led a **5-member engineering team**, delivering **3 key projects** on time.
      - Developed **automation tools** in **Python**, reducing manual effort by **30%**. 
      - Designed scalable **microservices architecture** using **FastAPI and Docker**.  
      - Optimized **data pipelines** for better real-time processing.  
      - Created **detailed API documentation**, improving developer adoption.     
        
        JOB DESCRIPTION CONTEXT:
        {job_description}
        
        ORIGINAL CONTENT:
        {original_content}
        **Ensure exactly two bullet points per job contain measurable outcomes.**   
        **Return ONLY the enhanced JSON for this section.**
        **Maintain the EXACT SAME structure but enhance the content.**
    """,
    
    "education": """
        Enhance the education section.
        
        INSTRUCTIONS:
        - CRITICAL: {education_description}
        - Each bullet point MUST be exactly 10-13 words.
        - Highlight relevant coursework and projects that align with target role.
        - Add academic achievements if applicable.
        - Keep institution names, degrees, and dates unchanged.
        
        JOB DESCRIPTION CONTEXT:
        {job_description}
        
        ORIGINAL CONTENT:
        {original_content}
        
        Return ONLY the enhanced JSON for this section.
        Maintain the EXACT SAME structure but enhance the content.
    """,
    
    "skills": """
        Enhance the skills section.
        
        INSTRUCTIONS:
        - CRITICAL: {skills_description}
        - Prioritize skills mentioned in the job description.
        - Organize skills in order of relevance to the position.
        - Each skill MUST be a specific, concise term (1-3 words maximum).
        - Add any relevant skills that may be missing based on experience.
        
        JOB DESCRIPTION CONTEXT:
        {job_description}
        
        ORIGINAL CONTENT:
        {original_content}
        
        Return ONLY the enhanced JSON for this section.
        Maintain the EXACT SAME structure but enhance the content.
    """,
    
    "projects": """
        Enhance the resume projects section by making it impactful and results-driven.  
        If a job description is provided, tailor the project descriptions to align with the relevant skills and requirements.  
        If a job description is NOT provided, improve the content while keeping it relevant and concise.
        
        INSTRUCTIONS:
        **STRICT RULES TO FOLLOW:** 
        - CRITICAL: {projects_description}
        - **Each bullet must be concise (28-32 words max).**  
        - **Each bullet must highlight key technologies, methodologies, or frameworks used.**  
        - **Include measurable outcomes in exactly TWO bullet points per project—no more.** 
        - **Ensure the remaining bullet points focus on skills, tools, and contributions WITHOUT measurable outcomes.**
        - **Do not force quantification on every point. Some points should highlight impact in qualitative terms.**

    **FORMAT EXAMPLES (Before & After):**  
      **Before:**  
      - Built a chatbot for customer service.  
      - Improved performance of ML models.  
      - Developed an API for data processing.  
      - Created a dashboard for analytics.  
      - Automated reporting tasks.  

      **After:**  
      - **Developed** a **customer service chatbot** using **Dialogflow and Python**, reducing query resolution time by **40%**.  
      - **Optimized** **ML models** using **XGBoost**, improving prediction accuracy by **15%**. 
      - **Designed** a **REST API** in **FastAPI**, enabling seamless data exchange.  
      - **Created** a **dashboard in Power BI**, enhancing real-time data visualization.  
      - **Automated** **reporting workflows**, reducing manual effort.  
        
        JOB DESCRIPTION CONTEXT:
        {job_description}
        
        ORIGINAL CONTENT:
        {original_content}
        
        **Ensure exactly two bullet points per project contain measurable outcomes.**  
        **Return ONLY the enhanced JSON for this section.**
        **Maintain the EXACT SAME structure but enhance the content.**
    """,
    
    "publications": """
        Enhance the publications section.
        
        INSTRUCTIONS:
        - Each publication description MUST be exactly 25-30 words.
        - Emphasize the relevance of the publication to the target position.
        - Highlight your specific contribution if multiple authors.
        - Use industry-specific terminology that aligns with the job.
        - Keep publication dates and formal details unchanged.
        
        JOB DESCRIPTION CONTEXT:
        {job_description}
        
        ORIGINAL CONTENT:
        {original_content}
        
        Return ONLY the enhanced JSON for this section.
        Maintain the EXACT SAME structure but enhance the content.
    """,
    
    "awards": """
        Enhance the awards section.
        
        INSTRUCTIONS:
        - CRITICAL: {awards_description}
        - Each award summary MUST be exactly 20-25 words.
        - Each award summary MUST include at least one metric or quantifiable achievement.
        - Emphasize the significance and exclusivity of each award.
        - Connect awards to specific achievements or skills.
        - Keep award names, issuers, and dates unchanged.
        
        JOB DESCRIPTION CONTEXT:
        {job_description}
        
        ORIGINAL CONTENT:
        {original_content}
        
        Return ONLY the enhanced JSON for this section.
        Maintain the EXACT SAME structure but enhance the content.
    """
}

def clean_llm_response(response_text: str) -> str:
    """Clean the LLM response by extracting and formatting valid JSON content using Pydantic."""
    try:
        # Initialize the parser with our model
        parser = PydanticOutputParser(pydantic_object=EnhancedResume)
        
        # Remove any markdown code blocks if present
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Try to parse the response directly
        try:
            parsed_obj = parser.parse(cleaned_text)
            return json.dumps(parsed_obj.dict(exclude_none=True), indent=2)
        except Exception as direct_parse_error:
            # If direct parsing fails, try to parse as raw JSON
            try:
                json_data = json.loads(cleaned_text)
                return json.dumps(json_data, indent=2)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON: {str(direct_parse_error)}")
                return "No valid JSON found"
    except Exception as e:
        logger.error(f"Error in clean_llm_response: {str(e)}")
        return "Invalid JSON format"

from langchain.output_parsers import PydanticOutputParser
def parse_json_safely(text: str) -> Dict:
    """Safely parse JSON using Pydantic models for validation."""
    try:
        # First try to parse as regular JSON
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}, text: {text[:100]}...")
        
        try:
            # Fix common JSON issues (replace single quotes with double quotes)
            text = text.replace("'", '"')
            
            # Try parsing again
            return json.loads(text)
        except Exception as second_error:
            logger.error(f"Secondary parsing error: {str(second_error)}")
            
            try:
                # As a last resort, try using the Pydantic parser
                parser = PydanticOutputParser(pydantic_object=EnhancedResume)
                parsed_obj = parser.parse(text)
                return parsed_obj.dict(exclude_none=True)
            except:
                logger.error("All parsing attempts failed")
                return {}

def get_template_params_for_section(section_name: str, template_type: str = "simple") -> Dict[str, str]:
    """
    Extract formatting parameters for a specific section from MCP template.
    
    Args:
        section_name: The name of the section (e.g., 'summary', 'experience')
        template_type: The template type to use (e.g., 'simple', 'software_engineer')
        
    Returns:
        Dict containing formatting parameters for the section
    """
    template_prompts = TEMPLATE_PROMPTS.get(template_type, TEMPLATE_PROMPTS['simple'])
    params = {}
    
    section_name_mapping = {
        "basics": "summary",
        "work": "experience"
    }
    
    mcp_section_name = section_name_mapping.get(section_name, section_name)
    
    if "parameters" in template_prompts:

        for param in template_prompts["parameters"]:
            if param["name"] == mcp_section_name and "description" in param:
                section_format_key = f"{mcp_section_name}_length"
                params[section_format_key] = param["description"]
                break
    
    return params

def create_section_prompt(
    section_name: str, 
    section_data: Any, 
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> str:
    """Create a prompt for enhancing a specific resume section using MCP template."""
    template_prompts = TEMPLATE_PROMPTS.get(template_type, TEMPLATE_PROMPTS['simple'])
    
    section_name_mapping = {
        "basics": "summary",
        "work": "experience"
    }
    
    mcp_section_name = section_name_mapping.get(section_name, section_name)
    
    if section_name in SECTION_PROMPTS:
        section_prompt = SECTION_PROMPTS[section_name]
        
        section_prompt = section_prompt.replace("{job_description}", job_description or "Not provided")
        section_prompt = section_prompt.replace("{original_content}", json.dumps(section_data, indent=2))
        
        for param in template_prompts.get("parameters", []):
            if param["name"] == mcp_section_name and "description" in param:
                description_placeholder = f"{{{mcp_section_name}_description}}"

                if description_placeholder in section_prompt:
                    section_prompt = section_prompt.replace(description_placeholder, param["description"])

                description_placeholder = f"{{{section_name}_description}}"
                if description_placeholder in section_prompt:
                    section_prompt = section_prompt.replace(description_placeholder, param["description"])
                

                length_placeholder = f"{{{mcp_section_name}_length}}"
                if length_placeholder in section_prompt:
                    section_prompt = section_prompt.replace(length_placeholder, param["description"])
                length_placeholder = f"{{{section_name}_length}}"
                if length_placeholder in section_prompt:
                    section_prompt = section_prompt.replace(length_placeholder, param["description"])
                
                format_placeholder = f"{{{mcp_section_name}_format}}"
                if format_placeholder in section_prompt:
                    section_prompt = section_prompt.replace(format_placeholder, param["description"])
                format_placeholder = f"{{{section_name}_format}}"
                if format_placeholder in section_prompt:
                    section_prompt = section_prompt.replace(format_placeholder, param["description"])
                
                break
        
        return section_prompt
    
    # If no section prompt template, use a direct approach with the parameter description
    # Find the parameter that matches this section name
    section_param = None
    for param in template_prompts.get("parameters", []):
        if param["name"] == mcp_section_name:
            section_param = param
            break
    
    if section_param and "description" in section_param:
        # Create a prompt specific to this section using the parameter description
        return f"""
            Enhance the {section_name} section according to these guidelines:
            
            STRICT RULES:
            {section_param['description']}
            
            JOB DESCRIPTION:
            {job_description or "Not provided"}
            
            ORIGINAL CONTENT:
            {json.dumps(section_data, indent=2)}
            
            Return ONLY the enhanced JSON for this section.
            Maintain the EXACT SAME structure but enhance the content.
        """
    
    # Generic fallback for sections without specific prompts or parameters
    return f"""
        Enhance the {section_name} section.
        
        INSTRUCTIONS:
        - Transform content to be more impactful and relevant to the target position.
        - Use strong, action-oriented language.
        - Add specific details and metrics where possible.
        
        JOB DESCRIPTION:
        {job_description or "Not provided"}
        
        ORIGINAL CONTENT:
        {json.dumps(section_data, indent=2)}
        
        Return ONLY the enhanced JSON for this section.
        Maintain the EXACT SAME structure but enhance the content.
    """

def extract_response_text(response: Any, model: BaseLanguageModel) -> str:
    """Extract the response text based on model type."""
    if isinstance(model, GoogleGenerativeAI):
        return response.text if hasattr(response, 'text') else str(response)
    elif isinstance(model, (ChatOpenAI, ChatDeepSeek, OpenAI)):
        return response.content if hasattr(response, 'content') else str(response)
    return str(response)

def get_token_counts_from_response(response: Any) -> Dict[str, int]:
    """Extract token counts from OpenAI model response via LangChain"""
    token_counts = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    

    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if hasattr(metadata, 'token_usage') or 'token_usage' in metadata:
            token_usage = metadata.get('token_usage', {})
            token_counts["input_tokens"] = token_usage.get('prompt_tokens', 0)
            token_counts["output_tokens"] = token_usage.get('completion_tokens', 0)
            token_counts["total_tokens"] = token_usage.get('total_tokens', 0)
    
    return token_counts

async def enhance_resume_section(
    section_name: str,
    section_data: Any,
    model: BaseLanguageModel,
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    try:
        logger.info(f"[DEBUG] Enhancing section: {section_name}")
        logger.info(f"[DEBUG] Section input: {section_data}")
        section_prompt = create_section_prompt(
            section_name, 
            section_data, 
            job_description,
            template_type
        )
        logger.info(f"[DEBUG] Section prompt for {section_name}: {section_prompt}")
        response = await model.ainvoke(section_prompt)
        response_text = extract_response_text(response, model)
        logger.info(f"[DEBUG] Section model response for {section_name}: {response_text}")
        cleaned_response = clean_llm_response(response_text)
        
        # Get token counts using our comprehensive function
        token_counts = get_token_counts_from_response(response)
        
        # Log token usage
        logger.info(f"Section: {section_name} | Input tokens: {token_counts['input_tokens']} | Output tokens: {token_counts['output_tokens']} | Total tokens: {token_counts['total_tokens']}")
        
        # Parse the enhanced content
        enhanced_section = parse_json_safely(cleaned_response)
        if not enhanced_section:
            logger.warning(f"Failed to enhance section {section_name}, keeping original")
            return {
                "section_data": section_data, 
                "input_tokens": token_counts['input_tokens'], 
                "output_tokens": token_counts['output_tokens'],
                "total_tokens": token_counts['total_tokens']
            }
            
        return {
            "section_data": enhanced_section, 
            "input_tokens": token_counts['input_tokens'], 
            "output_tokens": token_counts['output_tokens'],
            "total_tokens": token_counts['total_tokens']
        }
    except Exception as e:
        logger.error(f"Error enhancing section {section_name}: {e}")
        return {"section_data": section_data, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
async def enhance_resume_by_sections(
    json_data: Dict[str, Any],
    model: BaseLanguageModel,
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """Enhance each section of the resume asynchronously and combine results."""
    if 'details' in json_data and isinstance(json_data['details'], dict):
        resume_data = json_data['details']
        has_details_wrapper = True
    else:
        resume_data = json_data
        has_details_wrapper = False
    
    logger.info(f"Starting resume enhancement with template_type: {template_type}")
    
    
    # Convert template_type to lowercase for case-insensitive comparison
    template_type = template_type.lower()
    
    enhancement_tasks = {}
    for section_name, section_data in resume_data.items():
        task = asyncio.create_task(
            enhance_resume_section(
                section_name, 
                section_data, 
                model, 
                job_description,
                template_type
            )
        )
        enhancement_tasks[section_name] = task
    
    enhanced_sections = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    
    for section_name, task in enhancement_tasks.items():
        result = await task
        enhanced_sections[section_name] = result["section_data"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_tokens += result.get("total_tokens", 0)

    # Log total token usage
    logger.info(f"Resume enhancement completed | Template: {template_type} | "
                f"Total input tokens: {total_input_tokens} | "
                f"Total output tokens: {total_output_tokens} | "
                f"Total tokens: {total_tokens}")
    
    enhanced_resume = enhanced_sections
    
    if has_details_wrapper:
        result = {'details': enhanced_resume}
        if 'JD' in json_data:
            result['JD'] = json_data['JD']
        result['token_metrics'] = {
            'template_type': template_type,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens
        }
        return result
    
    # Add token metrics to the result
    enhanced_resume['token_metrics'] = {
        'template_type': template_type,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_tokens
    }
    
    return enhanced_resume

async def enhance_resume_with_model(
    json_data: Dict[str, Any],
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """
    Enhance resume data using the specified LLM model, section by section.
    
    Args:
        json_data: The resume data in JSON format
        job_description: Optional job description to tailor the resume
        template_type: Style template to use (simple or software_engineer)
    
    Returns:
        Dict containing the enhanced resume data with token metrics
    """
    try:
        model = SimpleModelManager().get_model()
        jd = job_description
        if not jd and 'JD' in json_data:
            jd = json_data.get('JD')
        
        # Check if Theme is in json_data and use it if template_type wasn't specified
        if template_type == "simple" and "Theme" in json_data:
            template_type = json_data["Theme"].lower()
            
        # Ensure template_type exists in TEMPLATE_PROMPTS
        if template_type not in TEMPLATE_PROMPTS:
            logger.warning(f"Template type '{template_type}' not found in TEMPLATE_PROMPTS. Using 'simple' instead.")
            template_type = "simple"
            
        # Log the template type being used
        logger.info(f"Using template_type: {template_type} for resume enhancement")
            
        enhanced_json = await enhance_resume_by_sections(
            json_data, 
            model, 
            jd,
            template_type
        )
        
        logger.info("Resume enhancement completed successfully")
        return enhanced_json
        
    except Exception as e:
        logger.error(f"Resume enhancement failed: {str(e)}")
        # Return original data with token metrics to ensure consistent response structure
        if 'token_metrics' not in json_data:
            json_data['token_metrics'] = {
                'template_type': template_type,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'error': str(e)
            }
        return json_data    

async def process_resume(
    json_data: Dict[str, Any],
    job_description: Optional[str] = None,
    template_type: str = "simple"
) -> Dict[str, Any]:
    """
    Process a resume by enhancing it section by section.
    
    Args:
        json_data: The resume data in JSON format
        job_description: Optional job description to tailor the resume
        template_type: Style template to use (simple or software_engineer)
        
    Returns:
        Dict containing the enhanced resume data
    """
    # Check for Theme in the JSON
    if "Theme" in json_data:
        template_type = json_data["Theme"].lower()
        logger.info(f"Using theme from JSON: {template_type}")
    
    enhanced_json = await enhance_resume_with_model(
        json_data,
        job_description,
        template_type
    )
    
    return enhanced_json

async def debug_resume_enhancement(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug wrapper for resume enhancement process to identify issues.
    
    Args:
        json_data: The resume data in JSON format
    
    Returns:
        Dict containing the enhanced resume data with debugging info
    """
    # 1. Extract theme/template_type
    template_type = "simple"  # Default
    if "Theme" in json_data:
        template_type = json_data["Theme"].lower()
        logger.info(f"Found Theme in JSON: {template_type}")
    
    # 2. Inspect template structure
    if template_type in TEMPLATE_PROMPTS:
        template = TEMPLATE_PROMPTS[template_type]
        logger.info(f"Template '{template_type}' found in TEMPLATE_PROMPTS")
        logger.info(f"Template structure: {list(template.keys())}")
        
        # Log parameters if available
        if "parameters" in template:
            logger.info("Template parameters:")
            for param in template["parameters"]:
                logger.info(f"  - {param.get('name', 'unnamed')}: {param.get('type', 'unknown type')}")
    else:
        logger.warning(f"Warning: Template '{template_type}' not found in TEMPLATE_PROMPTS")
        logger.info(f"Available templates: {list(TEMPLATE_PROMPTS.keys())}")
        template_type = "simple"  # Fallback
    
    # 3. Check if all required sections have parameters in the template
    if "details" in json_data and isinstance(json_data["details"], dict):
        resume_sections = list(json_data["details"].keys())
    else:
        resume_sections = list(json_data.keys())
        if "Theme" in resume_sections:
            resume_sections.remove("Theme")
        if "JD" in resume_sections:
            resume_sections.remove("JD")
    
    logger.info(f"Resume sections: {resume_sections}")
    
    if "parameters" in TEMPLATE_PROMPTS.get(template_type, {}):
        template_sections = [param["name"] for param in TEMPLATE_PROMPTS[template_type]["parameters"]]
        logger.info(f"Template sections: {template_sections}")
        
        # Check for missing sections
        section_mapping = {
            "basics": "summary",
            "work": "experience"
        }
        
        for section in resume_sections:
            mapped_section = section_mapping.get(section, section)
            if mapped_section not in template_sections:
                logger.warning(f"Warning: Section '{section}' (mapped to '{mapped_section}') not found in template parameters")
    
    # 4. Process the resume
    logger.info("Starting resume enhancement process...")
    try:
        enhanced_json = await enhance_resume_with_model(
            json_data=json_data,
            template_type=template_type
        )
        logger.info("Resume enhancement completed successfully")
        
        # 5. Add debugging info to the result
        if "token_metrics" in enhanced_json:
            enhanced_json["token_metrics"]["debug_info"] = {
                "template_used": template_type,
                "template_found": template_type in TEMPLATE_PROMPTS,
                "sections_processed": resume_sections
            }
        else:
            enhanced_json["debug_info"] = {
                "template_used": template_type,
                "template_found": template_type in TEMPLATE_PROMPTS,
                "sections_processed": resume_sections
            }
        
        return enhanced_json
    except Exception as e:
        logger.error(f"Error during enhancement: {str(e)}")
        # Return original with error info
        json_data["error"] = str(e)
        return json_data

