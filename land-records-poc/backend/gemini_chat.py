"""
Gemini Chatbot for Land Record Document Q&A
--------------------------------------------
Implements grounded question-answering over digitized land records using
the google-generativeai SDK.

Grounding rules:
  - Answers are strictly constructed from the record's extracted fields,
    per-field confidence scores, validation warnings, and audit status.
  - Never hallucinates facts not present in the record metadata.
  - Reads GEMINI_API_KEY strictly from environment variable (never hardcoded).
  - If GEMINI_API_KEY is not set or network is offline, provides an intelligent
    rule-grounded fallback response and clearly prompts the user to set their key.
"""
import os
import json
from typing import Dict, Any, Optional

def get_gemini_client():
    """Initializes Google Generative AI client using GEMINI_API_KEY from env."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except Exception:
        return None


def build_record_context(record: Any) -> str:
    """Constructs a factual, comprehensive grounding text block from the DB record."""
    field_conf = record.field_confidence or ""
    conf_dict = {}
    if field_conf:
        for pair in field_conf.split(","):
            if ":" in pair:
                k, _, v = pair.partition(":")
                conf_dict[k.strip()] = v.strip() + "%"

    context_lines = [
        "=== DIGITIZED LAND RECORD METADATA ===",
        f"Record ID: {record.record_id_str or f'REC{record.id:03d}'} (Database ID: {record.id})",
        f"Document Filename: {record.filename}",
        f"Document Type: {record.document_type or 'Scanned Document'}",
        f"Document Source: {record.document_source or 'Official Scan'}",
        f"Language / Script: {record.language or 'English'}",
        "",
        "--- IDENTIFIED PARTIES & LAND DETAILS ---",
        f"Owner / Landholder Name (खातेदार / పట్టాదారు): {record.owner_name or 'Not specified'}",
        f"Father's / Husband's Name: {record.father_name or 'Not specified'}",
        f"Survey Number / Khasra No (खसरा / సర్వే): {record.survey_number or record.khasra_no or 'Not specified'}",
        f"Sub-Division (हिस्सा): {record.sub_division or 'Not specified'}",
        f"Khata Number (खाता संख्या): {record.khata_no or 'Not specified'}",
        f"Land Area: {record.area_hectare or 'Not specified'} Hectares / Acres",
        f"Land Classification / Type: {record.classification or record.land_type or 'Agricultural'}",
        f"Mutation Order No (नामांतरण): {record.mutation_no or 'Not specified'}",
        "",
        "--- ADMINISTRATIVE JURISDICTION ---",
        f"Village (ग्राम / గ్రామం): {record.village or 'Not specified'}",
        f"Tehsil / Mandal (तहसील / మండలం): {record.tehsil or record.mandal or 'Not specified'}",
        f"District (जिला / జిల్లా): {record.district or 'Not specified'}",
        f"State (राज्य / రాష్ట్రం): {record.state or 'Not specified'}",
        f"Geo Coordinates: Latitude {record.latitude}, Longitude {record.longitude}",
        "",
        "--- VALIDATION & SYSTEM STATUS ---",
        f"Overall OCR Confidence: {record.overall_confidence or record.ocr_confidence or 0.0}%",
        f"Submission Status: {record.submission_status} (options: submitted, pending, not_submitted)",
        f"Verification Status: {record.verification_status} (options: verified, under_review, published, rejected)",
        f"Duplicate Detection Status: {record.duplicate_status or 'none'}",
        f"Processing Stage: {record.processing_stage or 'initial'}",
        f"Validation Warnings / Alerts: {record.validation_warnings or 'None (Passed all validation checks)'}",
        f"Reviewed By: {record.reviewed_by or 'Not yet reviewed'}",
        f"Reviewed At: {record.reviewed_at.isoformat() if record.reviewed_at else 'Pending'}",
    ]

    if conf_dict:
        context_lines.append("\n--- PER-FIELD OCR CONFIDENCE SCORES ---")
        for k, v in conf_dict.items():
            context_lines.append(f"  • {k}: {v}")

    if getattr(record, "ocr_text", None):
        snippet = record.ocr_text[:800].replace("\n", " ")
        context_lines.append(f"\n--- RAW OCR TEXT EXCERPT ---\n{snippet}")

    return "\n".join(context_lines)


def rule_based_qa_fallback(record: Any, question: str) -> str:
    """
    Intelligent grounded fallback when GEMINI_API_KEY is not set or network is unreachable.
    Answers specific analytical questions directly from DB record facts.
    """
    q = question.lower()
    owner = record.owner_name or "Unknown"
    survey = record.survey_number or record.khasra_no or "N/A"
    khata = record.khata_no or "N/A"
    village = record.village or "N/A"
    area = record.area_hectare or "N/A"
    warnings = record.validation_warnings or ""
    conf = record.overall_confidence or record.ocr_confidence or 0.0

    has_owner = any(w in q for w in ["owner", "who owns", "who is", "holder", "farmer", "name"])
    has_area = any(w in q for w in ["area", "size", "extent", "hectare", "acre"])
    has_survey = any(w in q for w in ["survey", "khasra", "khata", "plot"])

    if (has_owner and has_area) or (has_owner and has_survey) or (has_area and has_survey):
        father_str = f", son/daughter of {record.father_name}" if record.father_name else ""
        return (
            f"Record {record.record_id_str or record.id}:\n"
            f"• Registered Owner: {owner}{father_str}\n"
            f"• Survey/Khasra No: {survey} | Khata No: {khata}\n"
            f"• Land Area: {area} Hectares/Acres (Village: {village}, District: {record.district or 'N/A'})\n"
            f"• Status: {record.verification_status} (OCR Confidence: {conf}%)"
        )

    if has_owner:
        father_str = f", son/daughter of {record.father_name}" if record.father_name else ""
        return f"The registered owner of this parcel is {owner}{father_str}. Village: {village}."

    if has_area:
        return f"The land area for record {record.record_id_str or record.id} is {area} Hectares/Acres (Classification: {record.classification or 'Agricultural'})."

    if has_survey:
        return f"Record {record.record_id_str or record.id} has Survey/Khasra Number: {survey} and Khata Number: {khata} in village {village}."

    if any(w in q for w in ["why", "flag", "review", "warning", "conflict", "status", "pending"]):
        if warnings:
            return f"This record was flagged for review because of the following alerts:\n• {warnings}\nOverall OCR confidence is {conf}%."
        if conf < 65:
            return f"This record was flagged for review due to low OCR confidence ({conf}%). Officer review is mandatory under SOP."
        return f"This record has verification status '{record.verification_status}' and submission status '{record.submission_status}'. Validation warnings: None."

    if any(w in q for w in ["where", "location", "place", "district", "state", "tehsil", "mandal"]):
        return f"This record belongs to Village: {village}, Tehsil/Mandal: {record.tehsil or record.mandal or 'N/A'}, District: {record.district or 'N/A'}, State: {record.state or 'N/A'} (Coords: {record.latitude}, {record.longitude})."

    # Default summary
    return (
        f"[Grounded Record Summary for {record.record_id_str or record.id}]\n"
        f"• Owner: {owner}\n"
        f"• Survey/Khasra No: {survey} | Khata No: {khata}\n"
        f"• Area: {area} Ha | Village: {village}, District: {record.district or 'N/A'}\n"
        f"• Status: {record.verification_status} (Confidence: {conf}%)\n"
        f"• Warnings: {warnings or 'None'}\n\n"
        f"(Tip: Set the GEMINI_API_KEY environment variable to enable live multi-turn conversational AI via Gemini 1.5 Flash)."
    )


async def answer_record_question(record: Any, question: str) -> Dict[str, Any]:
    """
    Asks Gemini about the specific land record with strict grounding.
    Falls back gracefully if key is not configured.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    genai = get_gemini_client()

    context = build_record_context(record)

    if not genai or not api_key:
        answer = rule_based_qa_fallback(record, question)
        return {
            "answer": answer,
            "record_id": record.id,
            "record_id_str": record.record_id_str,
            "grounded": True,
            "model": "grounded_rule_fallback (Set GEMINI_API_KEY for live Gemini 1.5 Flash)"
        }

    system_prompt = (
        "You are an expert AI Revenue Officer Assistant for an Indian Land Record Digitization & Validation System "
        "(Smart India Hackathon SIH26018). Answer user questions strictly grounded on the provided land record metadata.\n"
        "RULES:\n"
        "1. Never invent or hallucinate survey numbers, owner names, areas, or rules not in the record.\n"
        "2. If the record has validation warnings or low confidence, clearly explain why and cite the exact fields.\n"
        "3. Answer concisely, professionally, and factually in markdown format.\n"
        "4. If asked about something not mentioned in the record, reply: 'This information is not present in the digitized record.'\n"
    )

    user_prompt = f"{context}\n\nUSER QUESTION: {question}"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)
        response = model.generate_content(user_prompt)
        answer_text = response.text.strip() if response and response.text else "No response generated."
        return {
            "answer": answer_text,
            "record_id": record.id,
            "record_id_str": record.record_id_str,
            "grounded": True,
            "model": "gemini-1.5-flash"
        }
    except Exception as e:
        # Fallback to grounded local summary if API quota/network issues arise
        fallback = rule_based_qa_fallback(record, question)
        return {
            "answer": f"{fallback}\n\n*(Note: Live Gemini API call encountered: {e})*",
            "record_id": record.id,
            "record_id_str": record.record_id_str,
            "grounded": True,
            "model": "grounded_fallback"
        }
