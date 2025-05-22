import json
import logging
from typing import Dict, Any
from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from .modelmanager import SimpleModelManager
from dotenv import load_dotenv
import re

# Load environment 
load_dotenv()
logger = logging.getLogger(__name__)

# Cache for analysis results
_analysis_cache = {
    'result': None,
    'resume': None,
    'job_description': None
}

# Extracts and cleans JSON content from model responses, handling different formats
def clean_json_response(response_text: str) -> str:
    """Extract and clean JSON from model response."""
    try:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.dumps(json.loads(json_str))
            except json.JSONDecodeError:
                logger.warning("Found JSON-like content in backticks, but failed to parse")
                
        json_match = re.search(r'(\{[\s\S]*\})', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.dumps(json.loads(json_str))
            except json.JSONDecodeError:
                logger.warning("Found JSON-like content, but failed to parse")
        
        logger.error(f"Could not extract JSON from response: {response_text[:500]}")
        return "{}"
    except Exception as e:
        logger.error(f"Error in clean_json_response: {str(e)}")
        return "{}"

 # Returns a standardized identifier string for the model type (gemini, openai, deepseek)   
def get_model_name(model: BaseLanguageModel) -> str:
    """Get the string identifier for the model type."""
    model_mapping = {
        GoogleGenerativeAI: "gemini",
        ChatOpenAI: "openai",
        ChatDeepSeek: "deepseek"
    }
    return model_mapping.get(type(model), "unknown")


# Extracts text content from different LLM response formats based on the model type
def extract_response_text(response: Any, model: BaseLanguageModel) -> str:
    """Extract the response text based on model type."""
    if isinstance(model, GoogleGenerativeAI):
        return response.text if hasattr(response, 'text') else str(response)
    elif isinstance(model, (ChatOpenAI, ChatDeepSeek)):
        return response.content if hasattr(response, 'content') else str(response)
    return str(response)

# Returns structured feedback with suggestions for improvement
async def analyze_resume(
    resume_json: Dict[str, Any],
    job_description: str
) -> Dict[str, Any]:
    """
    Analyze resume JSON against job description.
    
    Args:
        resume_json: The resume in JSON format
        job_description: The job description to analyze against
        
    Returns:
        Dict: Analysis results in JSON format
    """

    if (_analysis_cache['result'] and 
        _analysis_cache['resume'] == resume_json and 
        _analysis_cache['job_description'] == job_description):
        return _analysis_cache['result']

    try:
        model_manager = SimpleModelManager()
        model = model_manager.get_model()
        model_type = model_manager.current_model_type
        
        prompt = f"""
            You are an experienced Resume Evaluator Assistant specializing in providing detailed resume analysis and actionable feedback. Your task is to analyze resumes against specific job descriptions and provide comprehensive, structured feedback.
            In addition to suggestions, provide a short and concise recommendation (≤5 words) for each evaluation point. This short suggestion should be direct and to the point, making it easy for users to quickly understand key improvements without reading the full analysis.

            Resume JSON: {json.dumps(resume_json, indent=2)}
            Job Description: {job_description}

            Analyze and return results in this exact JSON format:
            {{
                "resume_evaluation_criteria": {{
                    "summary": {{
                        "relevance_to_job_description": {{
                            "analysis": "Highlight areas of expertise that are most relevant to the job description and align with the position",
                            "suggestion": "Align summary with JD key skills/technologies" 
                        }},
                        "results_and_impact": {{
                            "analysis": "Focus on concrete results or achievements that demonstrate how the candidate's contributions impacted organizations (e.g., increased revenue, improved efficiency)",
                            "suggestion": "Include measurable outcomes (e.g., 'increased X%')"
                        }},
                        "industry_experience": {{
                            "analysis": "Mention the types of organizations and industries the candidate has worked in and how this experience relates to the role they are applying for. Include the number of years of relevant experience in the field",
                            "suggestion": "Include measurable outcomes (e.g., 'increased X%')"
                        }},
                        "clarity_and_brevity": {{
                            "analysis": "Ensure the summary is clear, concise, and doesn't exceed one paragraph",
                            "suggestion": "Focus on 3–4 concise sentences highlighting strengths"
                        }},
                        "grammar_and_vocabulary": {{
                            "analysis": "Evaluate the summary for proper grammar and professional tone. Ensure correct sentence structure, proper use of punctuation, and an appropriate vocabulary level",
                            "suggestion": "Improve sentence structure and use professional language"
                        }},
                        "avoiding_generic_terms": {{
                            "analysis": "Avoid vague terms like 'results-driven,' 'team player,' and 'proven track record'",
                            "suggestion": "Replace vague terms with specific achievements (e.g., 'Led team to...')"
                        }}
                    }},
                    "education": {{
                        "relevance_to_job_description": {{
                            "analysis": "Assess whether the degree, coursework, and any additional educational qualifications align with the job description. Identify any missing relevant courses or specializations that could strengthen the candidate's profile",
                            "suggestion": "Highlight relevant degrees and coursework"
                        }},
                        "degree_institution_and_performance": {{
                            "analysis": "Verify that the degree, university name, and academic details (e.g., GPA, honors, or distinctions) are clearly mentioned and formatted correctly. Ensure the information is structured logically",
                            "suggestion": "State degree, institution, and GPA/percentage clearly"
                        }},
                        "grammar_spelling_and_consistency": {{
                            "analysis": "Perform a grammar and spelling check on the education section. Ensure consistency in formatting, degree naming conventions, and date representation. Identify any typos or inconsistencies",
                            "suggestion": "Ensure error-free, consistent formatting"
                        }}
                    }},
                    "skills": {{
                        "relevance_to_job_description": {{
                            "analysis": "Compare the skills listed in the Skills section with those required in the job description. Identify if the candidate's skills match the role and if any important skills from the JD are missing",
                            "suggestion": "Add missing JD skills"
                        }},
                        "technical_vs_interpersonal_balance": {{
                            "analysis": "Evaluate the balance between technical skills (e.g., programming languages, frameworks, databases, APIs) and interpersonal skills (e.g., problem-solving, collaboration, communication). Ensure the resume emphasizes the necessary technical skills for the role while also highlighting relevant interpersonal attributes such as leadership, collaboration, adaptability, and time management. If any critical skills are missing or underrepresented, provide suggestions for improvement",
                            "suggestion": "Balance technical, interpersonal, and leadership skills"
                        }},
                        "proficiency_level": {{
                            "analysis": "Assess if the candidate specifies the proficiency level of each skill. Check whether the proficiency levels match the expectations of the role",
                            "suggestion": "Include missing proficiency levels"
                        }},
                        "industry_relevant_skills": {{
                            "analysis": "Check if the candidate lists industry-specific skills that are essential for the job.These could include niche tools, technologies, or certifications that are important in the industry",
                            "suggestion": "Include industry-specific tools and methodologies"
                        }},
                        "skill_group_logic": {{
                            "analysis": "Assess if the skills listed are organized into logical groups or categories (e.g., programming languages, tools, frameworks, methodologies, and interpersonal skills). Ensure the structure is clear and easy to understand. Identify any skills that could be better categorized",
                            "suggestion": "Group skills by type (languages, tools)"
                        }},
                        "spell_check": {{
                            "analysis": "Perform a thorough spell check on the skills listed in the Skills section. Ensure all skills are spelled correctly and consistently. Identify any typos, misspellings, and common variations (e.g., 'Javascript' should be 'JavaScript')",
                            "suggestion": "Correct misspellings (e.g., 'Langauges' to 'Languages')"
                        }}
                    }},
                    "work": {{
                        "relevance_to_job_description": {{
                            "analysis": "Analyze the Experience section and compare it with the job description. Identify whether the responsibilities, projects, and tasks align with the role. Highlight any gaps where key job requirements are missing",
                            "suggestion": "Align experience with job role and responsibilities"
                        }},
                        "action_oriented_and_impactful": {{
                            "analysis": "Evaluate if each bullet point in the Experience section starts with a strong action verb. Check whether the descriptions demonstrate clear contributions, outcomes, and impact rather than just listing responsibilities",
                            "suggestion": "Use strong verbs, highlight contributions and impact"
                        }},
                        "tools_and_technologies": {{
                            "analysis": "Assess whether the mentioned tools, technologies, and frameworks align with the job description. Identify missing or outdated technologies relevant to the role",
                            "suggestion": "Include relevant tools, technologies, and frameworks"
                        }},
                        "career_growth_and_leadership": {{
                            "analysis": "Assess whether the candidate's experience shows career progression (e.g., promotions, increasing responsibilities) and leadership contributions (e.g., team management, project ownership, mentorship). Highlight areas where growth or leadership impact can be better demonstrated",
                            "suggestion": "Showcase leadership, promotions, and growth"
                        }},
                        "quantification_and_achievements": {{
                            "analysis": "Check if the Experience section includes measurable achievements such as percentages, revenue growth, cost savings, or efficiency improvements. Identify missing quantifiable data that strengthens impact",
                            "suggestion": "Use numbers to highlight quantifiable achievements"
                        }},
                        "grammar_and_language": {{
                            "analysis": "Check for grammar errors, clarity, and sentence structure in the Experience section. Ensure vocabulary is professional, precise, and free from redundancy or informal language",
                            "suggestion": "Improve clarity, grammar, and professional tone"
                        }}
                    }},
                    "projects": {{
                        "relevance_to_job_description_Technologies_Tools_and_Impact": {{
                            "analysis": "Ensure the projects listed are directly relevant to the job role and highlight the essential technologies, tools, and skills required. Check for measurable outcomes and impact (e.g., performance improvements, cost reductions)",
                            "suggestion": "Align projects with job requirements, highlight results"
                        }},
                        "action_problem_solving_and_Innovation": {{
                            "analysis": "Evaluate whether the project descriptions emphasize action-oriented tasks and innovative solutions. Look for problem-solving approaches and creative techniques used to overcome challenges",
                            "suggestion": "Use action verbs, showcase problem-solving"
                        }},
                        "clarity_structure_and_Duration": {{
                            "analysis": "Check how well the resume highlights collaboration, teamwork, and leadership experiences. Look for any indication of taking ownership or working in cross-functional teams",
                            "suggestion": "Emphasize leadership, teamwork, and collaboration"
                        }},
                        "collaboration_and_leadership": {{
                            "analysis": "Review if the projects are clearly structured with well-defined objectives, methodologies, and outcomes. Ensure the duration and timelines are included for better understanding of the project scope",
                            "suggestion": "Ensure clarity, include project timelines"
                        }},
                        "quantification_and_achievements": {{
                            "analysis": "Confirm that the project outcomes are quantified (e.g., percentages, savings, efficiency improvements) to showcase measurable achievements. Check for clear results or success indicators",
                            "suggestion": "Quantify results with measurable outcomes"
                        }},
                        "grammar_language_Accuracy_Vocabulary": {{
                            "analysis": "Check for grammar errors, clarity, and sentence structure in the Projects section. Ensure vocabulary is professional, precise, and free from redundancy or informal language.Ensure technical terms are used correctly and consistently",
                            "suggestion": "Maintain accurate grammar and consistent technical terms"
                        }}
                    }}
                }}
            }}

            Analysis Guidelines:
            - Compare the resume content with the job description and identify missing, irrelevant, or weak sections.
            - Focus on **alignment with JD**, **clarity**, **conciseness**, **grammar**, **professional language**, and **quantifiable impact**.
            - Provide **specific, actionable** suggestions for improvement in each section.
            - Avoid generic feedback; ensure recommendations are tailored to the job description and candidate's field.
            - Ensure skills, experience, and projects are formatted **logically** and **effectively highlight** the candidate's strengths.
            - Output the evaluation in **JSON format only**, without additional text.

            Example Output Structure:

                    {{
                "resume_evaluation": {{
                    "summary": {{
                        "relevance_to_job_description": {{
                            "analysis": "The summary does not highlight specific skills or tools mentioned in the JD, such as Generative AI, NLP, or Python libraries.",
                            "suggestion": "Align summary with JD skills: Generative AI, Python, NLP"
                        }}
                            
                    }},
                    "education": {{
                        "relevance_to_job_description": {{
                            "analysis": "The degrees (MSc in Data Science, BTech in Computer Science) align well with the JD requirement of a Bachelor's or Master's degree in Data Science, Computer Science, Statistics, Mathematics, or a related field.The relevant coursework includes Machine Learning and Deep Learning, which are beneficial for the statistical and ML techniques mentioned in the JD.However, there is no mention of SQL, Data Visualization (Power BI, Tableau), or statistical analysis coursework, which are key for the role.",
                            "suggestion": "Data Science, Computer Science, Machine Learning, Deep Learning"
                        }} 
                    }},
                    "skills": {{
                        "relevance_to_job_description": {{
                            "analysis": "Matches JD: React.js, Angular, MySQL, RESTful APIs, Docker. Missing Key Skills: Java, Spring Boot, Microservices, Kubernetes, Kafka, CI/CD.",
                            "suggestion": "Add Java, Spring Boot, Microservices, Kubernetes, Kafka, CI/CD."
                        }}   
                    }},
                    "work": {{
                        "relevance_to_job_description": {{
                            "analysis": "The experience section aligns well with the JD. The candidate's responsibilities like data analysis, SQL querying, and using Power BI are directly relevant to the tasks mentioned in the JD (data profiling, data transformation, data visualization, etc.). However, there could be more emphasis on the specific tools mentioned in the JD, such as Snowflake, dbt, or Azure DF.",
                            "suggestion": "Align experience with SQL, Snowflake, and data transformation." 
                        }}   
                    }},
                    "projects": {{
                        "relevance_to_job_description_Technologies_Tools_and_Impact": {{
                            "analysis": "Strong alignment with the JD; highlights the use of key tools (SQL, Excel, Power BI, Python) and showcases measurable impact (revenue improvement by 15%).",
                            "suggestion": "Well-aligned with job requirements."
                            }}    
                    }}
                }}
            }}
            Return only the JSON response without any additional text.
            """

        result = await model.ainvoke(prompt)
        response_text = extract_response_text(result, model)
        

        logger.debug(f"Raw response from LLM: {response_text[:1000]}")
        
        cleaned_response = clean_json_response(response_text)
        

        logger.debug(f"Cleaned JSON response: {cleaned_response[:1000]}")
        
        try:
            analysis_result = json.loads(cleaned_response)
            
            
            if not analysis_result or "resume_evaluation_criteria" not in analysis_result:
                logger.warning("Missing expected 'resume_evaluation_criteria' key in analysis result")
                
                analysis_result = {
                    "resume_evaluation_criteria": {
                        "error": {
                            "missing_data": {
                                "analysis": "The analysis did not return the expected data structure. Please try again.",
                                "suggestion": "Retry analysis"
                            }
                        }
                    }
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON: {str(e)}")
            analysis_result = {
                "error": f"Failed to parse analysis result: {str(e)}",
                "resume_evaluation_criteria": {}
            }

        
        _analysis_cache['result'] = analysis_result
        _analysis_cache['resume'] = resume_json
        _analysis_cache['job_description'] = job_description

    

        return analysis_result

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        
        return {"error": f"Analysis failed: {str(e)}"}


# Simplified version of analyze_resume that returns only detailed analysis without suggestions
async def deep_analysis(
    resume_json: Dict[str, Any],
    job_description: str
) -> Dict[str, Any]:
    """Get detailed analysis without suggestions."""
    try:

        full_analysis = await analyze_resume(resume_json, job_description)
        
        if "error" in full_analysis:
            return full_analysis

        deep_analysis = {"resume_evaluation_criteria": {}}
        
        for section, content in full_analysis.get("resume_evaluation_criteria", {}).items():
            deep_analysis["resume_evaluation_criteria"][section] = {}
            for subsection, details in content.items():
                deep_analysis["resume_evaluation_criteria"][section][subsection] = {
                    "analysis": details.get("analysis", "")
                }
                
        return deep_analysis
        
    except Exception as e:
        logger.error(f"Deep analysis failed: {str(e)}")
        return {"error": f"Deep analysis failed: {str(e)}"}

# Simplified version of analyze_resume that returns only brief suggestions without detailed analysis   
async def brief_analysis(
    resume_json: Dict[str, Any],
    job_description: str
) -> Dict[str, Any]:
    """Get brief analysis with only suggestions."""
    try:

        full_analysis = await analyze_resume(resume_json, job_description)
        
        if "error" in full_analysis:
            return full_analysis

        brief_analysis = {"resume_evaluation_criteria": {}}
        
        for section, content in full_analysis.get("resume_evaluation_criteria", {}).items():
            brief_analysis["resume_evaluation_criteria"][section] = {}
            for subsection, details in content.items():
                brief_analysis["resume_evaluation_criteria"][section][subsection] = {
                    "suggestion": details.get("suggestion", "")
                }
                
        return brief_analysis
        
    except Exception as e:
        logger.error(f"Brief analysis failed: {str(e)}")
        return {"error": f"Brief analysis failed: {str(e)}"}