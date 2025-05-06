from django.urls import path
from . import views

urlpatterns = [
    path('resumes/generate/', views.GenerateAIResumeView.as_view(), name='generate_ai_resume'),
    path('resumes/process/', views.ResumeProcessView.as_view(), name='process_resume'),
    path('resumes/extract/', views.ResumeExtractView.as_view(), name='extract_json'),
    path('resumes/generate/raw/', views.GenerateRawResume.as_view(), name='generate_raw_resume'),
    path('resumes/enhance-sentences/', views.SentenceEnhanceView.as_view(), name='enhance_sentences'),
    path('resumes/analyze/', views.AnalyzeResumeView.as_view(), name='analyze_resume'),
    path('resumes/analyze/deep/', views.DeepAnalysisView.as_view(), name='deep_analysis'),
    path('resumes/analyze/brief/', views.BriefAnalysisView.as_view(), name='brief_analysis'),
    path('resumes/enhance/typst/', views.EnhanceAndUpdateTypst.as_view(), name='enhance_and_update_typst'),
]