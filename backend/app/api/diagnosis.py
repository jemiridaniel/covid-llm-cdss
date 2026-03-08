import logging
from datetime import datetime
from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import PatientInput, DiagnosisResult, ReportRequest
from app.services.diagnosis_service import get_service
from app.services.llm_service import LLMService
from app.services.report import generate_report

router = APIRouter(tags=["diagnosis"])
llm_service = LLMService()
logger = logging.getLogger(__name__)


@router.post("/diagnose", response_model=DiagnosisResult)
async def diagnose(request: PatientInput):
    try:
        svc = get_service()
        result = svc.predict(request)

        llm_result = llm_service.explain(request, result["severity"])

        return DiagnosisResult(
            severity=result["severity"],
            confidence=result["confidence"],
            shap_features=result["shap_features"],
            llm_explanation=llm_result.clinical_reasoning,
            treatment_recommendation=result["treatment_recommendation"],
            model_used=llm_result.model_used,
            model_mode=result.get("model_mode", "mexican_only"),
            patient_name=request.patient_name,
            patient_id=request.patient_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def report(request: ReportRequest):
    try:
        pdf_bytes = generate_report(request.model_dump())
        patient_slug = request.patient_id or request.patient_name or "patient"
        safe_slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in patient_slug)
        filename = f"covid-report-{safe_slug}.pdf"
        return StreamingResponse(
            BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
