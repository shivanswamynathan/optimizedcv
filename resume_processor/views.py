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
                job_description = None
                user_id = json_data.get('user_id')
            else:
                json_data = json.loads(request.data.get('resume'))
                job_description = request.data.get('job_description')
                user_id = request.data.get('user_id')

            resume_data = json_data.get('details', json_data)
            theme_type = json_data.get('Theme', 'simple')
            token_metrics = {}

            if job_description or json_data.get('JD'):
                jd = job_description or json_data.get('JD')
                enhanced_data = await enhance_resume_with_model(
                    json_data=resume_data,
                    job_description=jd,
                    template_type=theme_type
                )
                token_metrics = enhanced_data.get('token_metrics', {})
                resume_data = enhanced_data

            # Store the enhanced resume in MongoDB
            mongo_service = MongoDBService()

            mongo_document = {
                'resume_data': resume_data,
                'theme_type': theme_type
            }

            # Add job description if available
            if job_description or json_data.get('JD'):
                mongo_document['job_description'] = job_description or json_data.get('JD')
                
            # Add token metrics if available
            if token_metrics:
                mongo_document['token_metrics'] = token_metrics
                
            # Insert document into MongoDB
            resume_id = mongo_service.insert_enhanced_resume(mongo_document, user_id)

            response_data = {
                "message": "Resume processed successfully",
                "data": resume_data,
                "token_metrics": token_metrics
            }
            
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