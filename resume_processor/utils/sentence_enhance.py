import json
import logging
from typing import Dict, Any, List, Optional
from .modelmanager import SimpleModelManager

logger = logging.getLogger(__name__)

def create_enhancement_prompt(sentence: str, context: Optional[str] = None, style: str = "professional") -> str:
    """
    Create a prompt for enhancing a specific sentence based on user-provided context and style.
    
    Args:
        sentence: The original sentence to enhance
        context: Optional additional context or specific enhancement instructions
        style: The enhancement style (professional, concise, impactful, etc.)
        
    Returns:
        str: The formatted prompt for the LLM
    """
    style_guidelines = {
        "professional": "Use formal language, industry terminology, and focus on achievements with metrics.",
        "concise": "Make every word count. Aim for brevity while maintaining clarity and impact.",
        "impactful": "Focus on results, achievements, and quantifiable outcomes. Use strong action verbs.",
        "technical": "Emphasize technical skills, tools, and methodologies. Use industry-specific terminology.",
        "leadership": "Highlight leadership qualities, team management, and strategic initiatives."
    }
    
    # Default to professional if style not found
    style_guide = style_guidelines.get(style.lower(), style_guidelines["professional"])
    
    return f"""Enhance the following resume sentence to make it more {style} and impactful.

ORIGINAL SENTENCE:
{sentence}

ENHANCEMENT STYLE:
{style_guide}

SPECIFIC CONTEXT/INSTRUCTIONS:
{context or "No specific context provided. Focus on general enhancement."}

GUIDELINES:
- Maintain factual accuracy and original meaning
- Use specific, quantifiable metrics where possible (or preserve existing ones)
- Use strong action verbs
- Avoid generic claims without supporting evidence
- Keep approximately the same length as the original
- Do not add completely new information that wasn't implied in the original

Your task is to rewrite the sentence to make it more effective, professional, and impactful.

Return ONLY the enhanced sentence, without explanations or formatting.
"""

async def enhance_sentence(
    sentence: str, 
    context: Optional[str] = None, 
    style: str = "professional"
) -> Dict[str, Any]:
    """
    Enhance a single sentence using the configured LLM.
    
    Args:
        sentence: The original sentence to enhance
        context: Optional additional context or specific enhancement instructions
        style: The enhancement style (professional, concise, impactful, etc.)
        
    Returns:
        Dict containing the original and enhanced sentences
    """
    try:
        if not sentence or not sentence.strip():
            return {
                "original": sentence,
                "enhanced": sentence,
                "error": "Empty sentence provided"
            }
        
        model_manager = SimpleModelManager()
        model = model_manager.get_model()
        
        prompt = create_enhancement_prompt(sentence, context, style)
        
        response = await model.ainvoke(prompt)
        tokens = response.response_metadata.get('token_usage', {}).get('total_tokens')

        # Extract the response based on model type
        if hasattr(response, 'content'):
            enhanced_sentence = response.content
        elif hasattr(response, 'text'):
            enhanced_sentence = response.text
        else:
            enhanced_sentence = str(response)
        
        # Clean up the response
        enhanced_sentence = enhanced_sentence.strip()
        
        # Remove any markdown or code formatting
        enhanced_sentence = enhanced_sentence.replace('```', '').strip()
        
        # Log the enhancement for analysis
        logger.info(f"Enhanced sentence. Original: '{sentence}' -> Enhanced: '{enhanced_sentence}'")
        
        return {
            "original": sentence,
            "enhanced": enhanced_sentence,
            "token_usage":tokens
        }
        
    except Exception as e:
        logger.error(f"Error enhancing sentence: {str(e)}")
        return {
            "original": sentence,
            "enhanced": sentence,
            "error": str(e)
        }

async def enhance_multiple_sentences(
    sentences: List[str],
    context: Optional[str] = None,
    style: str = "professional"
) -> List[Dict[str, Any]]:
    """
    Enhance multiple sentences in parallel.
    
    Args:
        sentences: List of sentences to enhance
        context: Optional additional context or specific enhancement instructions
        style: The enhancement style (professional, concise, impactful, etc.)
        
    Returns:
        List of dictionaries containing original and enhanced sentences
    """
    try:
        import asyncio
        
        if not sentences:
            return []
        
        tasks = []
        for sentence in sentences:
            task = asyncio.create_task(
                enhance_sentence(sentence, context, style)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
        
    except Exception as e:
        logger.error(f"Error enhancing multiple sentences: {str(e)}")
        return [
            {"original": sentence, "enhanced": sentence, "error": str(e)}
            for sentence in sentences
        ]