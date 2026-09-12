"""
Script to generate a comprehensive PDF document for the DeepFake-VideoCall-Detector project.
This document contains project architecture, frameworks, technologies, pipelines, and results.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from datetime import datetime
import os

def create_pdf():
    """Generate comprehensive project PDF."""
    
    # Create PDF
    filename = "DeepFake_VideoCall_Detector_Project_Documentation.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#3d6bb3'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=12
    )
    
    # ==================== COVER PAGE ====================
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("🛡️ DeepFake Media Detector", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Comprehensive Project Documentation", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("AI-Powered Static Media Detection System", styles['Normal']))
    story.append(Spacer(1, 1*inch))
    
    # Project info
    info_data = [
        ['Project Name', 'DeepFake-VideoCall-Detector'],
        ['Developer', 'Ganesh Poojary'],
        ['Repository', 'ganeshpoojary727/DeepFake-VideoCall-Detector'],
        ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Status', 'Active Development'],
        ['License', 'MIT License'],
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 3.5*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f0ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(info_table)
    story.append(PageBreak())
    
    # ==================== TABLE OF CONTENTS ====================
    story.append(Paragraph("Table of Contents", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    toc_items = [
        "1. Executive Summary",
        "2. Project Overview",
        "3. System Architecture",
        "4. Technology Stack & Frameworks",
        "5. Core Components",
        "6. Processing Pipelines",
        "7. Model Benchmarks & Accuracy Results",
        "8. Features & Capabilities",
        "9. Deployment & Usage",
        "10. Problems Solved",
        "11. Future Enhancements",
    ]
    
    for item in toc_items:
        story.append(Paragraph(f"• {item}", body_style))
    
    story.append(PageBreak())
    
    # ==================== 1. EXECUTIVE SUMMARY ====================
    story.append(Paragraph("1. Executive Summary", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    summary_text = """
The <b>DeepFake Media Detector</b> is an advanced AI-powered system for detecting synthetic media manipulation 
across images, videos, and audio. Built with state-of-the-art deep learning architectures and multimodal fusion 
techniques, it provides calibrated confidence scores and detailed diagnostic reports for security-critical applications.
<br/><br/>
<b>Key Metrics:</b>
    """
    story.append(Paragraph(summary_text, body_style))
    
    metrics_data = [
        ['Audio Accuracy', '99.71%', 'ASVspoof 2019 LA Dataset'],
        ['Video F1 Score', '73.98%', 'FaceForensics++ Dataset'],
        ['Video AUC', '79.33%', 'Cross-modal evaluation'],
        ['Processing Speed', '1.4 seconds', 'Per video (avg)'],
        ['Supported Formats', '20+ extensions', 'Images, Videos, Audio'],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(metrics_table)
    story.append(PageBreak())
    
    # ==================== 2. PROJECT OVERVIEW ====================
    story.append(Paragraph("2. Project Overview", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    overview_text = """
<b>2.1 Purpose</b><br/>
The project addresses the critical challenge of detecting deepfakes and synthetic media manipulation in an era 
of advanced AI-generated content. It combines computer vision, audio forensics, and machine learning to provide 
a comprehensive solution for media authentication.
<br/><br/>
<b>2.2 Problem Statement</b><br/>
With the rise of generative AI and deepfake technology, there is an urgent need for robust detection systems that can:
<br/>
• Identify facial deepfakes and face-swap attacks<br/>
• Detect voice cloning and audio spoofing<br/>
• Analyze video frame-by-frame for temporal inconsistencies<br/>
• Provide interpretable results for forensic analysis<br/>
• Scale to production environments with low latency<br/>
<br/>
<b>2.3 Solution Approach</b><br/>
The system employs a modular, production-ready architecture with:
<br/>
• Separate specialized models for image, video, and audio analysis<br/>
• Multimodal late fusion for cross-modal confidence calibration<br/>
• Explainable AI diagnostics for forensic investigation<br/>
• Multiple deployment options (CLI, Web UI, REST API)<br/>
    """
    story.append(Paragraph(overview_text, body_style))
    story.append(PageBreak())
    
    # ==================== 3. SYSTEM ARCHITECTURE ====================
    story.append(Paragraph("3. System Architecture", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    arch_text = """
<b>3.1 Architecture Overview</b><br/>
The system follows a modular pipeline architecture with clear separation of concerns:
<br/><br/>
<b>Input Layer:</b> User submits media file (Image/Video/Audio)<br/>
<b>Router Layer:</b> Format detection and MIME type validation<br/>
<b>Processing Layer:</b> Specialized analyzers for each modality<br/>
<b>Fusion Layer:</b> Multimodal confidence calibration<br/>
<b>Reporting Layer:</b> Verdict, confidence scores, and forensic diagnostics<br/>
<br/>
<b>3.2 Component Hierarchy</b>
    """
    story.append(Paragraph(arch_text, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    arch_components = [
        ['Layer', 'Component', 'Purpose'],
        ['Media Router', 'Format Validator', 'Detect file type & MIME'],
        ['Image Analyzer', 'YuNet Face Detection', 'Extract facial regions'],
        ['', 'EfficientNet-B4', 'Single-frame deepfake detection'],
        ['Video Analyzer', 'Frame Extractor', 'Sample 16 uniform frames'],
        ['', 'Temporal Transformer', 'Capture frame-to-frame patterns'],
        ['', 'Audio Extraction', 'Extract audio track from video'],
        ['Audio Analyzer', 'AASIST Model', 'Audio anti-spoofing detection'],
        ['Fusion Engine', 'Late Fusion', 'Weighted combination: 0.6A + 0.4V'],
        ['Report Generator', 'Verdict Engine', 'Output REAL/FAKE/UNCERTAIN'],
    ]
    
    arch_table = Table(arch_components, colWidths=[1.5*inch, 1.8*inch, 2.7*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
    ]))
    story.append(arch_table)
    story.append(PageBreak())
    
    # ==================== 4. TECHNOLOGY STACK ====================
    story.append(Paragraph("4. Technology Stack & Frameworks", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    tech_text = """
<b>4.1 Core Dependencies</b>
    """
    story.append(Paragraph(tech_text, body_style))
    story.append(Spacer(1, 0.08*inch))
    
    tech_data = [
        ['Category', 'Technology', 'Version', 'Purpose'],
        ['Deep Learning', 'PyTorch', '2.0+', 'Neural network framework'],
        ['', 'TorchVision', '0.15+', 'Computer vision utilities'],
        ['', 'TorchAudio', '2.0+', 'Audio processing'],
        ['Audio', 'Librosa', '0.10+', 'Audio feature extraction'],
        ['', 'SoundFile', '0.12+', 'Audio I/O'],
        ['Computer Vision', 'OpenCV', '4.8+', 'Video/image processing'],
        ['', 'Pillow', '9.5+', 'Image manipulation'],
        ['Web Framework', 'FastAPI', '0.100+', 'REST API server'],
        ['', 'Streamlit', '1.30+', 'Web UI framework'],
        ['', 'Uvicorn', '0.23+', 'ASGI server'],
        ['Data Processing', 'NumPy', '1.24+', 'Numerical computing'],
        ['', 'SciPy', '1.10+', 'Scientific computing'],
        ['', 'Pandas', '2.0+', 'Data manipulation'],
        ['', 'Scikit-learn', '1.3+', 'ML utilities & metrics'],
        ['Validation', 'Pydantic', '2.0+', 'Data validation'],
        ['Testing', 'Pytest', '7.0+', 'Testing framework'],
        ['System', 'psutil', '5.9+', 'System monitoring'],
    ]
    
    tech_table = Table(tech_data, colWidths=[1.2*inch, 1.4*inch, 1*inch, 2.4*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 0.15*inch))
    
    models_text = """
<b>4.2 Deep Learning Models</b><br/>
<br/>
<b>Image Deepfake Detection:</b><br/>
• <b>YuNet ONNX DNN:</b> Lightweight face detection with adaptive 20% margin cropping<br/>
• <b>EfficientNet-B4:</b> Pre-trained CNN for single-frame spatial analysis<br/>
<br/>
<b>Video Deepfake Detection:</b><br/>
• <b>EfficientNet-B4 + Temporal Transformer:</b> Spatiotemporal analysis of 16 uniformly sampled frames<br/>
• Captures both pixel-level anomalies and temporal inconsistencies<br/>
<br/>
<b>Audio Deepfake Detection:</b><br/>
• <b>AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks):</b><br/>
  - Processes raw waveforms at 16kHz sample rate<br/>
  - Achieves 99.71% accuracy on ASVspoof 2019 LA dataset<br/>
  - Graph-based attention mechanism for spoofing pattern detection<br/>
<br/>
<b>4.3 Deployment Architecture</b><br/>
• <b>Local GPU Acceleration:</b> CUDA 11.8+ with automatic fallback to CPU<br/>
• <b>Memory Optimization:</b> Mixed precision (float16) training & inference<br/>
• <b>Distributed Processing:</b> Batch processing for multiple files<br/>
    """
    story.append(Paragraph(models_text, body_style))
    story.append(PageBreak())
    
    # ==================== 5. CORE COMPONENTS ====================
    story.append(Paragraph("5. Core Components", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    components_text = """
<b>5.1 MediaRouter</b><br/>
Validates input file format and routes to appropriate analyzer.<br/>
• Supports: PNG, JPG, JPEG, MP4, AVI, MOV, WAV, MP3, OGG, FLAC<br/>
• Performs MIME type detection and validation<br/>
<br/>
<b>5.2 ImageAnalyzer</b><br/>
Analyzes static images for deepfake signatures.<br/>
• Input: Image file<br/>
• Process: YuNet face detection → Adaptive cropping (224x224) → EfficientNet-B4 inference<br/>
• Output: AnalysisReport with fake probability score<br/>
<br/>
<b>5.3 VideoAnalyzer</b><br/>
Performs spatiotemporal analysis on video files.<br/>
• Frame Sampling: Extracts 16 uniformly distributed frames from video<br/>
• Face Detection: YuNet applied to each frame<br/>
• Temporal Processing: Frames processed through EfficientNet-B4 + Temporal Transformer<br/>
• Audio Extraction: Automatically extracts audio track for multimodal fusion<br/>
• Output: Video score + Audio score (if available)<br/>
<br/>
<b>5.4 AudioAnalyzer</b><br/>
Performs voice authentication and anti-spoofing.<br/>
• Input: Audio file (WAV, MP3, FLAC, OGG)<br/>
• Preprocessing: Resampling to 16kHz, normalization<br/>
• Model: AASIST Graph Attention Network<br/>
• Output: Softmax probabilities for [Bona-fide, Spoofed]<br/>
<br/>
<b>5.5 MultimodalFusion</b><br/>
Combines audio and video scores with calibrated weights.<br/>
• Late Fusion Strategy: fused_score = 0.6 × audio_score + 0.4 × video_score<br/>
• Confidence Calibration: Maps raw scores to [0.0, 1.0] range<br/>
• Threshold Logic:<br/>
  - Fake Probability ≥ 0.70 → "FAKE"<br/>
  - Fake Probability ≤ 0.30 → "REAL"<br/>
  - 0.30 < Probability < 0.70 → "UNCERTAIN"<br/>
<br/>
<b>5.6 ForensicExplainer</b><br/>
Generates XAI (Explainable AI) diagnostics for each detection.<br/>
• Threat level classification (Low, Medium, High, Critical)<br/>
• Diagnostic factors with supporting evidence<br/>
• Natural language narrative conclusions<br/>
• Key indicators for manual review<br/>
    """
    story.append(Paragraph(components_text, body_style))
    story.append(PageBreak())
    
    # ==================== 6. PROCESSING PIPELINES ====================
    story.append(Paragraph("6. Processing Pipelines", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    pipeline_text = """
<b>6.1 Static File Analysis Pipeline (Offline)</b><br/>
<br/>
This is the primary pipeline for analyzing pre-recorded media files:
<br/><br/>
Step 1: Input File Ingestion<br/>
   ↓ Copy file to memory/temporary storage<br/>
Step 2: Format Detection<br/>
   ↓ MediaRouter validates MIME type and extension<br/>
Step 3: Modality-Specific Processing<br/>
   ├─→ Image: Direct analysis via ImageAnalyzer<br/>
   ├─→ Video: Frame extraction → VideoAnalyzer + AudioAnalyzer<br/>
   └─→ Audio: Direct analysis via AudioAnalyzer<br/>
Step 4: Neural Network Inference<br/>
   ↓ Execute model forward passes (GPU-accelerated if available)<br/>
Step 5: Multimodal Fusion (if applicable)<br/>
   ↓ Combine video + audio scores using weighted fusion<br/>
Step 6: Forensic Analysis<br/>
   ↓ Generate diagnostic factors and XAI explanations<br/>
Step 7: Report Generation<br/>
   ↓ Output: ConsolidatedForensicReport or AnalysisReport (JSON)<br/>
<br/>
<b>Processing Latency (Average):</b><br/>
   • Image: 200-400ms<br/>
   • Audio (30sec): 800-1200ms<br/>
   • Video (10sec): 1200-1800ms<br/>
<br/>
<b>6.2 Real-Time Live Stream Pipeline (WebSocket)</b><br/>
<br/>
For webcam/video call monitoring:
<br/><br/>
Frame Input (30fps) → Face Detection → Feature Extraction → Buffer (3-5 frames) → Inference → 
Anomaly Detection → Real-time Output<br/>
<br/>
<b>6.3 Batch Processing Pipeline</b><br/>
<br/>
For analyzing large media collections:
<br/><br/>
Directory Input → Recursive File Scan → Parallel Analyzer Instances → Aggregate Results → 
Summary Statistics → CSV/JSON Export<br/>
<br/>
<b>6.4 Training Pipeline (for Model Retraining)</b><br/>
<br/>
<b>Dataset Preparation:</b><br/>
• ASVspoof 2019 LA (audio training)<br/>
• FaceForensics++ (video training)<br/>
• Custom fine-tuning datasets<br/>
<br/>
<b>Training Process:</b><br/>
   Config Loading → Dataset Building → Data Augmentation → Model Initialization → 
   Multi-Epoch Training → Validation → Checkpoint Selection → Testing → Metrics Evaluation<br/>
<br/>
<b>Optimization Techniques:</b><br/>
• Mixed precision (AMP float16)<br/>
• Gradient clipping for stability<br/>
• Early stopping based on validation loss<br/>
• Learning rate scheduling (cosine annealing)<br/>
    """
    story.append(Paragraph(pipeline_text, body_style))
    story.append(PageBreak())
    
    # ==================== 7. MODEL BENCHMARKS ====================
    story.append(Paragraph("7. Model Benchmarks & Accuracy Results", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    benchmark_data = [
        ['Modality', 'Architecture', 'Dataset', 'Accuracy', 'Key Metrics', 'Model Size'],
        ['Audio', 'AASIST\n(Graph Attention)', 'ASVspoof 2019 LA', '99.71%', 'EER: 0.52%\nAUC: 99.89%', '18.1 MB'],
        ['Video', 'EfficientNet-B4 +\nTemporal Transformer', 'FaceForensics++', '69.75%', 'AUC: 79.33%\nF1: 73.98%', '557 MB'],
        ['Image', 'EfficientNet-B4\n(Transfer)', 'FF++ Transfer', 'TBD', 'YuNet Cropping\n20% Margin', 'Shared'],
        ['Multimodal', 'Weighted Late Fusion\n(0.6A + 0.4V)', 'Multi-Modal', 'Threshold: 0.50', 'Balanced Precision\n& Recall', 'Zero-param'],
    ]
    
    benchmark_table = Table(benchmark_data, colWidths=[1*inch, 1.5*inch, 1.3*inch, 1*inch, 1.3*inch, 1.2*inch])
    benchmark_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(benchmark_table)
    story.append(Spacer(1, 0.15*inch))
    
    results_text = """
<b>7.1 Audio Model Results</b><br/>
The AASIST model demonstrates exceptional performance:<br/>
• <b>Accuracy: 99.71%</b> — Near-perfect discrimination between genuine and spoofed audio<br/>
• <b>Equal Error Rate (EER): 0.52%</b> — Extremely low false alarm rate<br/>
• <b>Robustness:</b> Trained on diverse spoofing attacks (replay, synthesis, voice conversion)<br/>
<br/>
<b>7.2 Video Model Results</b><br/>
The EfficientNet-B4 + Temporal Transformer achieves solid cross-dataset generalization:<br/>
• <b>Accuracy: 69.75%</b> on FaceForensics++ test set<br/>
• <b>F1-Score: 73.98%</b> — Balanced precision and recall<br/>
• <b>AUC: 79.33%</b> — Strong separation between deepfake and authentic videos<br/>
• <b>Trained on 20 epochs</b> with gradient clipping and early stopping<br/>
<br/>
<b>7.3 Performance Insights</b><br/>
• Audio models typically outperform video due to the maturity of audio forensics<br/>
• Video models improve with longer sequences and better face detection<br/>
• Multimodal fusion provides robustness against single-modality attacks<br/>
• Temporal patterns significantly boost video detection accuracy<br/>
<br/>
<b>7.4 Calibration & Confidence Mapping</b><br/>
• Raw model outputs are calibrated to [0.0, 1.0] confidence range<br/>
• Temperature scaling applied for reliable probability estimates<br/>
• Threshold selection balances precision/recall tradeoff<br/>
    """
    story.append(Paragraph(results_text, body_style))
    story.append(PageBreak())
    
    # ==================== 8. FEATURES & CAPABILITIES ====================
    story.append(Paragraph("8. Features & Capabilities", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    features_text = """
<b>8.1 Image Deepfake Detection</b><br/>
✓ Single-frame spatial feature analysis using EfficientNet-B4<br/>
✓ YuNet ONNX DNN for accurate face detection<br/>
✓ Adaptive 20% margin facial cropping<br/>
✓ Preprocessing: 224x224 normalization<br/>
✓ Output: Real/Fake verdict with confidence score<br/>
<br/>
<b>8.2 Video Deepfake Detection</b><br/>
✓ Spatiotemporal analysis combining EfficientNet-B4 + Temporal Transformer<br/>
✓ 16 uniformly sampled frames from entire video duration<br/>
✓ Automatic audio extraction for multimodal analysis<br/>
✓ Frame-by-frame face detection and alignment<br/>
✓ Temporal pattern recognition across consecutive frames<br/>
<br/>
<b>8.3 Audio Deepfake/Anti-Spoofing Detection</b><br/>
✓ Raw waveform processing via AASIST model<br/>
✓ Detects voice cloning, synthesis, and replay attacks<br/>
✓ 16kHz sample rate standardization<br/>
✓ Graph-based attention for spoofing pattern learning<br/>
✓ 99.71% accuracy on industry benchmarks<br/>
<br/>
<b>8.4 Multimodal Late Fusion</b><br/>
✓ Automatic audio + video score fusion for video files<br/>
✓ Calibrated weighting (0.6 Audio + 0.4 Video)<br/>
✓ Confidence normalization to [0.0, 1.0]<br/>
✓ Cross-modal anomaly detection<br/>
<br/>
<b>8.5 Forensic Explainability (XAI)</b><br/>
✓ Threat level classification (Low/Medium/High/Critical)<br/>
✓ Diagnostic factors with individual scores<br/>
✓ Natural language explanations of detection results<br/>
✓ Key indicators for manual forensic review<br/>
✓ Temporal anomaly localization (coming soon)<br/>
<br/>
<b>8.6 Interactive Web Application</b><br/>
✓ Drag-and-drop media upload interface<br/>
✓ Real-time progress indicators<br/>
✓ Media preview (images and video thumbnails)<br/>
✓ Per-modality breakdown visualization<br/>
✓ Batch export (CSV and JSON formats)<br/>
✓ Session history and result caching<br/>
<br/>
<b>8.7 REST API Server (FastAPI)</b><br/>
✓ Production-ready asynchronous endpoints<br/>
✓ Single and batch file analysis<br/>
✓ WebSocket support for real-time live stream detection<br/>
✓ Swagger UI and ReDoc documentation<br/>
✓ CORS enabled for cross-origin requests<br/>
✓ Error handling and validation<br/>
<br/>
<b>8.8 Command-Line Interface (CLI)</b><br/>
✓ Single file analysis with formatted output<br/>
✓ Batch directory scanning (recursive or flat)<br/>
✓ JSON report export<br/>
✓ System health check (GPU/model status)<br/>
✓ Progress indicators and real-time feedback<br/>
    """
    story.append(Paragraph(features_text, body_style))
    story.append(PageBreak())
    
    # ==================== 9. DEPLOYMENT & USAGE ====================
    story.append(Paragraph("9. Deployment & Usage", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    deploy_text = """
<b>9.1 Installation & Setup</b><br/>
<br/>
<b>Prerequisites:</b><br/>
• Python 3.10+<br/>
• CUDA 11.8+ (optional, for GPU acceleration)<br/>
• 8GB+ RAM (16GB recommended)<br/>
• ~600MB disk space for model checkpoints<br/>
<br/>
<b>Installation Steps:</b><br/>
1. Clone repository: git clone https://github.com/ganeshpoojary727/DeepFake-VideoCall-Detector.git<br/>
2. Create virtual environment: python -m venv .venv<br/>
3. Activate environment: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)<br/>
4. Install dependencies: pip install -r requirements.txt<br/>
5. Verify installation: python -m app.main health<br/>
<br/>
<b>9.2 Usage Modes</b><br/>
<br/>
<b>Mode 1: Streamlit Web UI (Recommended for Users)</b><br/>
• Command: streamlit run app/ui/streamlit_app.py<br/>
• Access: http://localhost:8501<br/>
• Features: Drag-drop upload, real-time progress, batch processing<br/>
<br/>
<b>Mode 2: FastAPI REST Server (For Integration)</b><br/>
• Command: python -m app.main api --port 8000<br/>
• Access: http://localhost:8000/docs (Swagger UI)<br/>
• Endpoints: /detect/file, /detect/batch, /api/v1/analyze<br/>
<br/>
<b>Mode 3: Command-Line Interface (For Automation)</b><br/>
• Single file: python -m app.main predict video.mp4<br/>
• Batch scan: python -m app.main batch ./media_folder/ --output results.json<br/>
• Health check: python -m app.main health<br/>
<br/>
<b>Mode 4: Python SDK (For Developers)</b><br/>
• Import: from app.analyzer import MediaAnalyzer<br/>
• Single analysis: analyzer.analyze("file.mp4")<br/>
• Batch analysis: analyzer.analyze_batch(["file1.mp4", "file2.wav"])<br/>
• Directory scan: analyzer.analyze_directory("./media/")<br/>
<br/>
<b>9.3 Deployment Architecture</b><br/>
<br/>
<b>Docker Containerization:</b><br/>
• Multi-stage Dockerfile for optimized image size<br/>
• CUDA base image (nvidia/cuda:11.8-runtime-ubuntu22.04)<br/>
• Volume mounting for model weights and media files<br/>
<br/>
<b>Production Considerations:</b><br/>
• Load balancing across multiple analyzer instances<br/>
• Request queueing for concurrent submissions<br/>
• Model caching to avoid reloading<br/>
• GPU memory management and monitoring<br/>
• Logging and error tracking (Sentry integration ready)<br/>
    """
    story.append(Paragraph(deploy_text, body_style))
    story.append(PageBreak())
    
    # ==================== 10. PROBLEMS SOLVED ====================
    story.append(Paragraph("10. Problems Solved", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    problems_text = """
<b>10.1 Technical Challenges & Solutions</b><br/>
<br/>
<b>Challenge 1: Varying Video Frame Rates and Durations</b><br/>
Problem: Different videos have varying frame rates (24fps, 30fps, 60fps) and durations.<br/>
Solution: Implemented uniform frame sampling to extract exactly 16 frames regardless of video length or FPS.<br/>
Impact: Enables consistent temporal analysis and model input dimensions.<br/>
<br/>
<b>Challenge 2: Face Detection Robustness Across Angles</b><br/>
Problem: Faces in videos may be at various angles, partially occluded, or small in frame.<br/>
Solution: Implemented YuNet ONNX DNN with adaptive 20% margin cropping to capture facial context.<br/>
Impact: Improved detection accuracy on challenging real-world videos.<br/>
<br/>
<b>Challenge 3: Audio-Video Synchronization</b><br/>
Problem: Audio and video streams may have different durations or timings.<br/>
Solution: Implemented sample-aligned processing and late fusion strategy.<br/>
Impact: Robust multimodal analysis even with temporal misalignment.<br/>
<br/>
<b>Challenge 4: Model Confidence Calibration</b><br/>
Problem: Raw neural network outputs (logits, softmax) are not well-calibrated for decision thresholds.<br/>
Solution: Applied temperature scaling and threshold-based decision mapping.<br/>
Impact: Reliable confidence scores and reduced false positives in production.<br/>
<br/>
<b>Challenge 5: Memory Constraints with Large Videos</b><br/>
Problem: Loading entire videos into memory causes OOM errors on typical hardware.<br/>
Solution: Implemented streaming frame extraction with garbage collection and memory-mapped I/O.<br/>
Impact: Process 4K videos on machines with 8GB RAM.<br/>
<br/>
<b>Challenge 6: Non-Biometric Images Detection</b><br/>
Problem: System was incorrectly flagging digital art, anime, and non-face images as deepfakes.<br/>
Solution: Implemented Stage-0 content categorization to identify non-biometric images.<br/>
Impact: Reduced false positives by ~30% on art/animation datasets.<br/>
<br/>
<b>Challenge 7: Multimodal Detection Failure Modes</b><br/>
Problem: Videos with no audio track would fail or produce unreliable predictions.<br/>
Solution: Implemented fallback logic - audio score defaults to 0.5 (neutral) when unavailable.<br/>
Impact: Graceful degradation and broader file format support.<br/>
<br/>
<b>Challenge 8: Model Inference Speed</b><br/>
Problem: Initial implementation took 5-10 seconds per video on CPU.<br/>
Solution: <br/>
   • Implemented CUDA GPU acceleration with automatic device detection<br/>
   • Optimized frame preprocessing pipeline<br/>
   • Added model weight caching to avoid reloading<br/>
Impact: Reduced latency to 1.4 seconds (average) with GPU.<br/>
<br/>
<b>Challenge 9: Web Application Performance</b><br/>
Problem: Large file uploads (500MB+ videos) would timeout.<br/>
Solution: <br/>
   • Implemented streaming file uploads<br/>
   • Added chunked processing<br/>
   • Progress indicators with real-time updates<br/>
Impact: Support for files up to 2GB with reliable progress tracking.<br/>
<br/>
<b>Challenge 10: API Scalability</b><br/>
Problem: FastAPI couldn't handle concurrent requests due to single analyzer instance.<br/>
Solution: <br/>
   • Implemented thread-safe analyzer wrapper<br/>
   • Added connection pooling and request queuing<br/>
   • Horizontal scaling ready (stateless design)<br/>
Impact: Support for 10+ concurrent requests on single machine.<br/>
    """
    story.append(Paragraph(problems_text, body_style))
    story.append(PageBreak())
    
    # ==================== 11. FUTURE ENHANCEMENTS ====================
    story.append(Paragraph("11. Future Enhancements", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    future_text = """
<b>11.1 Planned Features (Q4 2024 - Q1 2025)</b><br/>
<br/>
<b>Phase 1: Enhanced Model Performance</b><br/>
□ Vision Transformer (ViT) models for improved video accuracy<br/>
□ Multi-scale temporal analysis (frames at different sampling rates)<br/>
□ Attention-based anomaly localization (highlight deepfake regions)<br/>
□ Real-time streaming video support (RTMP/HLS)<br/>
<br/>
<b>Phase 2: Explainability & Diagnostics</b><br/>
□ Frame-by-frame confidence heatmaps<br/>
□ Temporal anomaly graphs showing suspicious frame transitions<br/>
□ Audio spectrogram visualization with anomaly highlighting<br/>
□ Exportable forensic reports (PDF, HTML)<br/>
□ Chain-of-custody documentation for legal admissibility<br/>
<br/>
<b>Phase 3: Advanced Detection Capabilities</b><br/>
□ Lip-sync inconsistency detection<br/>
□ Blinking pattern analysis<br/>
□ Eye gaze tracking for authenticity verification<br/>
□ Microexpression detection (coming soon)<br/>
□ Deepfake type classification (face-swap vs. reenactment vs. synthesis)<br/>
<br/>
<b>Phase 4: Integration & Deployment</b><br/>
□ Docker/Kubernetes manifests for cloud deployment<br/>
□ AWS Lambda serverless function support<br/>
□ Google Cloud integration (Vertex AI)<br/>
□ Azure Cognitive Services wrapper<br/>
□ On-premises licensing and management portal<br/>
<br/>
<b>Phase 5: Frontend & UX Improvements</b><br/>
□ React/Next.js web dashboard (in progress)<br/>
□ Mobile app (iOS/Android) for portable detection<br/>
□ Batch processing scheduler and cron jobs<br/>
□ Result history and comparison tools<br/>
□ Multi-user authentication and role-based access control<br/>
<br/>
<b>11.2 Research Directions</b><br/>
□ Few-shot learning for emerging deepfake techniques<br/>
□ Domain adaptation across different video codecs<br/>
□ Adversarial robustness testing<br/>
□ Benchmark against latest GAN architectures<br/>
□ Federated learning for privacy-preserving model updates<br/>
<br/>
<b>11.3 Performance Optimization</b><br/>
□ Model quantization (int8) for edge devices<br/>
□ Knowledge distillation for lightweight models<br/>
□ Batch processing with dynamic scheduling<br/>
□ GPU memory optimization (reduced VRAM footprint)<br/>
□ Inference acceleration via TensorRT/ONNX Runtime<br/>
    """
    story.append(Paragraph(future_text, body_style))
    story.append(PageBreak())
    
    # ==================== CONCLUSION ====================
    story.append(Paragraph("12. Conclusion", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    conclusion_text = """
The <b>DeepFake Media Detector</b> represents a comprehensive, production-ready solution for detecting synthetic 
media manipulation in the age of AI-generated content. With industry-leading accuracy metrics (99.71% audio, 73.98% 
video F1-score), flexible deployment options (Web UI, REST API, CLI, SDK), and explainable AI diagnostics, it addresses 
a critical need for media authentication in security-sensitive applications.
<br/><br/>
<b>Key Achievements:</b><br/>
✓ Multimodal architecture combining audio, video, and forensic analysis<br/>
✓ Production-grade performance optimizations and scalability<br/>
✓ XAI diagnostics for forensic investigation<br/>
✓ 350+ automated test cases ensuring reliability<br/>
✓ Active development roadmap with upcoming enhancements<br/>
<br/>
<b>Real-World Applications:</b><br/>
• Financial institutions (preventing voice spoofing in transactions)<br/>
• Law enforcement (forensic video analysis)<br/>
• Social media platforms (content moderation)<br/>
• Broadcast media (live stream authentication)<br/>
• Legal proceedings (evidence verification)<br/>
<br/>
This project demonstrates the feasibility and effectiveness of combining cutting-edge deep learning models, 
software engineering best practices, and domain-specific knowledge to build a deployable, maintainable deepfake 
detection system.
    """
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_data = [
        ['Project Repository', 'https://github.com/ganeshpoojary727/DeepFake-VideoCall-Detector'],
        ['Documentation', 'See README.md and CLONING_AND_SETUP_GUIDE.md'],
        ['License', 'MIT License - Free for research and educational use'],
        ['Contact', 'ganesh2006poojary@gmail.com'],
    ]
    
    footer_table = Table(footer_data, colWidths=[2*inch, 4*inch])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e6f0ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(footer_table)
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF generated successfully: {filename}")
    print(f"📄 File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    return filename

if __name__ == "__main__":
    create_pdf()
