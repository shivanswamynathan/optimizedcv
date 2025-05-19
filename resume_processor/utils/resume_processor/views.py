from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from asgiref.sync import sync_to_async, async_to_sync
import logging  
from django.http import HttpResponse
import json
import base64
from .utils.enhance import enhance_resume_with_model
from .utils.extract import convert_pdf_to_json_schema
from .utils.sentence_enhance import enhance_multiple_sentences
from .utils.analysis import analyze_resume, deep_analysis, brief_analysis
from .utils.mongodb import MongoDBService
import asyncio
logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class GenerateRawResume(APIView):
    def post(self, request):
        return async_to_sync(self._async_post)(request)

    async def _async_post(self, request):
        try:
            if request.content_type == 'application/json':
                json_data = request.data
                job_description = None
            else:
                json_data = json.loads(request.data.get('resume'))
                job_description = request.data.get('job_description')

            jd = job_description or json_data.get('JD')
            if jd and str(jd).strip().lower() != "none":
                enhanced_data = await enhance_resume_with_model(json_data=json_data, job_description=jd)
                resume_data = enhanced_data
            else:
                resume_data = json_data


            return Response({
                "message": "Resume processed successfully",
                "data": resume_data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Error processing JSON: %s", str(e))
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class GenerateAIResumeView(APIView):
    def post(self, request):
        return async_to_sync(self._async_post)(request)

    async def _async_post(self, request):
        try:
            if request.content_type == 'application/json':
                json_data = request.data
                job_description = json_data.get('JD')
                user_id = json_data.get('user_id')
            else:
                json_data = json.loads(request.data.get('resume'))
                job_description = request.data.get('job_description') or json_data.get('JD')
                user_id = request.data.get('user_id')

            resume_data = json_data.get('details', json_data)
            token_metrics = {}

            if job_description is None or str(job_description).strip().lower() != "none":
                jd = None if job_description and str(job_description).strip().lower() == "none" else job_description

                enhanced_data = await enhance_resume_with_model(
                    json_data=resume_data,
                    job_description=jd
                )
                token_metrics = enhanced_data.get('token_metrics', {})
                
                # Extract enhancement suggestions BEFORE removing from enhanced_data
                enhancement_suggestions = enhanced_data.pop('enhancement_suggestions', [])
                
                logger.info(f"DEBUG: Extracted {len(enhancement_suggestions)} suggestions from enhanced_data: {enhancement_suggestions}")
                resume_data = enhanced_data
            else:
                enhancement_suggestions = []  # No suggestions if no enhancement

            # 2. Store the enhanced resume in MongoDB (without suggestions)
            mongo_service = MongoDBService()

            mongo_document = {
                'resume_data': resume_data
            }

            # Add job description if available
            if job_description and str(job_description).strip().lower() != "none":
                mongo_document['job_description'] = job_description
                
            # Add token metrics if available
            if token_metrics:
                mongo_document['token_metrics'] = token_metrics
            
            logger.info(f"About to insert document with keys: {list(mongo_document.keys())}")
                
            # Insert document into MongoDB and get the resume_id
            resume_id = mongo_service.insert_enhanced_resume(mongo_document, user_id)

            if resume_id:
                logger.info(f"Successfully inserted document with ID: {resume_id}")
                
                # 3. Store enhancement suggestions separately if they exist
                if enhancement_suggestions:
                    try:
                        logger.info(f"Processing {len(enhancement_suggestions)} enhancement suggestions")
                        
                        # Convert suggestions to the expected format for storage
                        questions_format = [{
                            'question': sug.get('suggestion', ''),
                            'category': sug.get('category', ''), 
                            'purpose': sug.get('improvement_made', ''),
                            'further_action': sug.get('further_action', '')
                        } for sug in enhancement_suggestions]
                        
                        questions_stored = mongo_service.insert_suggestion_questions(
                            questions_format, 
                            resume_id
                        )
                        
                        if questions_stored:
                            logger.info(f"Enhancement suggestions stored successfully for resume_id: {resume_id}")
                        else:
                            logger.warning(f"Failed to store enhancement suggestions for resume_id: {resume_id}")
                    except Exception as e:
                        logger.error(f"Error storing enhancement suggestions: {str(e)}")
                        logger.error(f"Enhancement suggestions format: {enhancement_suggestions}")
                else:
                    logger.info("No enhancement suggestions to store")
                        
            else:
                logger.warning("Failed to insert document into MongoDB")

            # 4. Prepare response
            response_data = {
                "message": "Resume processed successfully",
                "data": resume_data,
                "token_metrics": token_metrics
            }

            # Add enhancement suggestions to response (for immediate display)
            if enhancement_suggestions:
                response_data["enhancement_suggestions"] = enhancement_suggestions
            
            # Add resume_id to response if insertion was successful
            if resume_id:
                response_data["resume_id"] = resume_id
                
            return Response(response_data, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error("Error processing JSON: %s", str(e))
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class ResumeProcessView(APIView):
    def post(self, request):
        return async_to_sync(self._async_post)(request)

    async def _async_post(self, request):
        try:
            if 'resume' not in request.FILES:
                return Response(
                    {"error": "No resume file provided"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            resume_file = request.FILES['resume']
            job_description = request.data.get('job_description')
            pdf_content = await sync_to_async(resume_file.read)()
            json_data = await convert_pdf_to_json_schema(pdf_content)
            if "error" in json_data:
                return Response(
                    {"error": json_data["error"]}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            enhanced_json = await enhance_resume_with_model(
                json_data=json_data,
                job_description=job_description
            )
            # Convert to RenderCV schema
            if 'details' in enhanced_json:
                converted = enhanced_json['details']
            else:
                converted = enhanced_json
                
            return Response({
                "message": "Resume processed successfully",
                "data": converted
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Error processing resume: %s", str(e))
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@method_decorator(csrf_exempt, name='dispatch')
class ResumeExtractView(APIView):
    def post(self, request, *args, **kwargs):
        return async_to_sync(self._async_post)(request, *args, **kwargs)

    async def _async_post(self, request, *args, **kwargs):
        try:
         
            if 'resume' not in request.FILES:
                return Response(
                    {"error": "No resume file provided"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            resume_file = request.FILES['resume']
            pdf_content = await sync_to_async(resume_file.read)()
            
            json_resume = await convert_pdf_to_json_schema(
                pdf_content                
            )
            
            if "error" in json_resume:
                return Response(
                    {"error": json_resume["error"]}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            return Response({
                "message": "Resume extracted successfully",
                "data": json_resume
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error extracting resume JSON: {str(e)}")
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@method_decorator(csrf_exempt, name='dispatch')
class SentenceEnhanceView(APIView):
    def post(self, request):
        return async_to_sync(self._async_post)(request)

    async def _async_post(self, request):
        try:
            # Validate request data
            if request.content_type == 'application/json':
                data = request.data
            else:
                return Response(
                    {"error": "Content type must be application/json"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check for required fields
            if 'sentences' not in data:
                return Response(
                    {"error": "The 'sentences' field is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            sentences = data.get('sentences', [])
            context = data.get('context', None)
            style = data.get('style', 'professional')
            
            # Validate sentences format
            if isinstance(sentences, str):
                # Single sentence as string
                sentences = [sentences]
            elif not isinstance(sentences, list):
                return Response(
                    {"error": "The 'sentences' field must be a string or an array of strings"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            
            # Process the sentences
            results = await enhance_multiple_sentences(
                sentences=sentences,
                context=context,
                style=style
            )
            
            return Response({
                "message": "Sentences enhanced successfully",
                "results": results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error enhancing sentences: {str(e)}")
            return Response(
                {"error": str(e)},
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
@method_decorator(csrf_exempt, name='dispatch')
class SuggestionQuestionsView(APIView):
    """
    API endpoint to retrieve suggestion questions/enhancement suggestions by resume ID.
    """
    
    def get(self, request, resume_id=None):
        """
        GET method to retrieve suggestion questions for a specific resume.
        URL: /resumes/suggestion-questions/{resume_id}/
        """
        return async_to_sync(self._async_get)(request, resume_id)
    
    def post(self, request):
        """
        POST method to retrieve suggestion questions using resume_id in body.
        URL: /resumes/suggestion-questions/
        Body: {"resume_id": "60d21b4667d0d8992e610c85"}
        """
        return async_to_sync(self._async_post)(request)

    async def _async_get(self, request, resume_id):
        """Async GET handler for suggestion questions"""
        try:
            if not resume_id:
                return Response(
                    {"error": "Resume ID is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get questions from MongoDB
            mongo_service = MongoDBService()
            questions = mongo_service.get_suggestion_questions_by_resume_id(resume_id)
            
            if questions is None:
                return Response(
                    {"error": f"No suggestion questions found for resume ID: {resume_id}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            return Response({
                "message": "Suggestion questions retrieved successfully",
                "data": {
                    "resume_id": resume_id,
                    "suggestion_questions": questions
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in SuggestionQuestionsView GET: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    async def _async_post(self, request):
        """Async POST handler for suggestion questions"""
        try:
            # Handle JSON request body
            if request.content_type == 'application/json':
                data = request.data
            else:
                try:
                    data = json.loads(request.body)
                except json.JSONDecodeError:
                    return Response(
                        {"error": "Invalid JSON in request body"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            resume_id = data.get('resume_id')
            
            if not resume_id:
                return Response(
                    {"error": "Resume ID is required in request body"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Get questions from MongoDB
            mongo_service = MongoDBService()
            questions = mongo_service.get_suggestion_questions_by_resume_id(resume_id)
            
            if questions is None:
                return Response(
                    {"error": f"No suggestion questions found for resume ID: {resume_id}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            return Response({
                "message": "Suggestion questions retrieved successfully",
                "data": {
                    "resume_id": resume_id,
                    "suggestion_questions": questions
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in SuggestionQuestionsView POST: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
@method_decorator(csrf_exempt, name='dispatch')
class AnalyzeResumeView(APIView):
    def post(self, request, *args, **kwargs):
        return async_to_sync(self._async_post)(request, *args, **kwargs)

    async def _async_post(self, request, *args, **kwargs):
        try:
            # Handle both PDF file and JSON input
            if 'resume' in request.FILES:
                # Process PDF file
                resume_file = request.FILES['resume']
                pdf_content = await sync_to_async(resume_file.read)()
                
                # Convert PDF to JSON
                resume_json = await convert_pdf_to_json_schema(
                    pdf_content
                )
                
                if "error" in resume_json:
                    return Response(
                        {"error": resume_json["error"]}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            else:
                # Handle JSON input
                if request.content_type == 'application/json':
                    resume_json = request.data.get('resume')
                else:
                    resume_json = json.loads(request.data.get('resume'))
            
            job_description = request.data.get('job_description', '')
            
            # Ensure job_description is not None
            if job_description is None:
                job_description = ''
                
            logger.info(f"Analyzing resume against job description of length: {len(job_description)}")

            analysis_result = await analyze_resume(
                resume_json=resume_json,
                job_description=job_description
            )

            return Response({
                "message": "Analysis completed successfully",
                "data": analysis_result
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@method_decorator(csrf_exempt, name='dispatch')
class DeepAnalysisView(APIView):
    def post(self, request, *args, **kwargs):
        return async_to_sync(self._async_post)(request, *args, **kwargs)

    async def _async_post(self, request, *args, **kwargs):
        try:
            if 'resume' in request.FILES:
                resume_file = request.FILES['resume']
                pdf_content = await sync_to_async(resume_file.read)()
                
                resume_json = await convert_pdf_to_json_schema(
                    pdf_content
                )
                
                if "error" in resume_json:
                    return Response(
                        {"error": resume_json["error"]}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            else:
                if request.content_type == 'application/json':
                    resume_json = request.data.get('resume')
                else:
                    resume_json = json.loads(request.data.get('resume'))
            
            job_description = request.data.get('job_description')

            analysis_result = await deep_analysis(
                resume_json=resume_json,
                job_description=job_description
            )

            return Response({
                "message": "Deep analysis completed successfully",
                "data": analysis_result
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error performing deep analysis: {str(e)}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@method_decorator(csrf_exempt, name='dispatch')
class BriefAnalysisView(APIView):
    def post(self, request, *args, **kwargs):
        return async_to_sync(self._async_post)(request, *args, **kwargs)

    async def _async_post(self, request, *args, **kwargs):
        try:
            if 'resume' in request.FILES:
                resume_file = request.FILES['resume']
                pdf_content = await sync_to_async(resume_file.read)()
                
                resume_json = await convert_pdf_to_json_schema(
                    pdf_content
                )
                
                if "error" in resume_json:
                    return Response(
                        {"error": resume_json["error"]}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            else:
                if request.content_type == 'application/json':
                    resume_json = request.data.get('resume')
                else:
                    resume_json = json.loads(request.data.get('resume'))
            
            job_description = request.data.get('job_description')

            analysis_result = await brief_analysis(
                resume_json=resume_json,
                job_description=job_description
            )

            return Response({
                "message": "Brief analysis completed successfully",
                "data": analysis_result
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error performing brief analysis: {str(e)}")
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
@method_decorator(csrf_exempt, name='dispatch')
class SentenceEnhanceView(APIView):
    def post(self, request):
        return async_to_sync(self._async_post)(request)

    async def _async_post(self, request):
        try:
            # Validate request data
            if request.content_type == 'application/json':
                data = request.data
            else:
                return Response(
                    {"error": "Content type must be application/json"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check for required fields
            if 'sentences' not in data:
                return Response(
                    {"error": "The 'sentences' field is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            sentences = data.get('sentences', [])
            context = data.get('context', None)
            style = data.get('style', 'professional')
            
            # Validate sentences format
            if isinstance(sentences, str):
                # Single sentence as string
                sentences = [sentences]
            elif not isinstance(sentences, list):
                return Response(
                    {"error": "The 'sentences' field must be a string or an array of strings"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            
            # Process the sentences
            results = await enhance_multiple_sentences(
                sentences=sentences,
                context=context,
                style=style
            )
            
            return Response({
                "message": "Sentences enhanced successfully",
                "results": results
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error enhancing sentences: {str(e)}")
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )