import os
import re
import string
from pathlib import Path
from django.shortcuts import render
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "model"

model_path = MODEL_DIR / "fake_job_model.pkl"
vectorizer_path = MODEL_DIR / "vectorizer.pkl"

model = None
vectorizer = None
model_loaded = False

try:
    if model_path.exists() and vectorizer_path.exists():
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        model_loaded = True
        print("Model loaded successfully")
    else:
        print("Model files not found")
except Exception as e:
    print(f"Error loading model: {e}")


def safe_parse_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return default
        try:
            return int(value)
        except ValueError:
            return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def clean_input(text):
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) == 0:
        return None
    return text


SUSPICIOUS_KEYWORDS = {
    "+2 points": {
        "urgent hiring": 2,
        "immediate hiring": 2,
        "apply now": 2,
        "limited seats": 2,
        "hiring immediately": 2,
        "last chance": 2,
        "act fast": 2,
        "dont miss": 2,
        "time limited": 2,
        "only few": 2,
        "few spots": 2,
    },
    "+3 points": {
        "no experience": 3,
        "no experience needed": 3,
        "high salary": 3,
        "work from home": 3,
        "work from anywhere": 3,
        "home based": 3,
        "remote job": 3,
        "entry level": 3,
        "freshers welcome": 3,
        "students welcome": 3,
        "no degree": 3,
        "no diploma": 3,
        "weekly payment": 3,
        "daily payment": 3,
        "fast payment": 3,
    },
    "+5 points": {
        "guaranteed income": 5,
        "guaranteed salary": 5,
        "easy money": 5,
        "quick money": 5,
        "fast money": 5,
        "make money": 5,
        "no interview": 5,
        "no interview required": 5,
        "send bank details": 5,
        "bank account": 5,
        "wire transfer": 5,
        "western union": 5,
        "money gram": 5,
        "send money": 5,
        "processing fee": 5,
        "registration fee": 5,
        "joining fee": 5,
        "upfront payment": 5,
        "pay first": 5,
        "envelope stuffing": 5,
        "assembly work": 5,
        "certificate processing": 5,
        "data entry jobs": 5,
        "typing jobs": 5,
        "online jobs": 5,
    },
    "+4 points (missing info)": {
        "missing company": 4,
        "unknown company": 4,
        "confidential company": 4,
        "secret company": 4,
    },
}


def detect_keywords(text):
    cleaned_text = clean_input(text)
    if cleaned_text is None:
        return {
            "categories": {
                "+2 points": [],
                "+3 points": [],
                "+5 points": [],
                "+4 points (missing info)": [],
            },
            "total_score": 0,
        }

    text_lower = cleaned_text.lower()
    categories = {
        "+2 points": [],
        "+3 points": [],
        "+5 points": [],
        "+4 points (missing info)": [],
    }
    total_score = 0

    for category, keywords in SUSPICIOUS_KEYWORDS.items():
        for keyword, weight in keywords.items():
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text_lower):
                categories[category].append({"keyword": keyword, "weight": weight})
                total_score += weight

    return {"categories": categories, "total_score": total_score}


def calculate_risk_score(text):
    cleaned_text = clean_input(text)
    if cleaned_text is None:
        return {
            "keyword_score": 0,
            "salary_score": 0,
            "company_score": 0,
            "total_score": 0,
            "risk_level": "Low",
            "salary_mentioned": False,
            "unrealistic_salary": False,
            "company_present": True,
        }

    text_lower = cleaned_text.lower()
    keyword_result = detect_keywords(cleaned_text)
    keyword_score = keyword_result["total_score"]

    salary_score = 0
    salary_mentioned = False
    unrealistic_salary = False

    salary_patterns = [
        r"\$[\d,]+(?:-\$[\d,]+)?(?:\s*(?:per|/)\s*(?:month|year|annum))?",
        r"[\d,]+(?:-\d+,)*\d*\s*(?:k|K)?\s*(?:per|/)\s*(?:month|year|annum)",
        r"(?:salary|income|pay|earn)[\s:]*\$?[\d,]+(?:-\$?[\d,]+)?",
    ]

    for pattern in salary_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            salary_mentioned = True
            for match in matches:
                numbers = re.findall(r"(\d{4,})", match)
                for num in numbers:
                    num_int = safe_parse_int(num, 0)
                    if num_int > 20000 and (
                        "no experience" in text_lower or "entry level" in text_lower
                    ):
                        salary_score += 4
                        unrealistic_salary = True

    company_score = 0
    company_present = True

    company_red_flags = [
        r"confidential",
        r"unnamed",
        r"unknown",
        r"reputable company",
        r"major company",
        r"top company",
        r"leading company",
        r"fortune 500",
        r"no company",
        r"company name",
        r"company undisclosed",
    ]

    for flag in company_red_flags:
        if re.search(r"\b" + flag + r"\b", text_lower):
            company_score = 4
            company_present = False
            break

    base_risk_score = keyword_score + salary_score + company_score
    max_score = 25
    normalized_score = min(base_risk_score, max_score)

    if normalized_score >= 10:
        risk_level = "high"
    elif normalized_score >= 5:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "keyword_score": keyword_score,
        "salary_score": salary_score,
        "company_score": company_score,
        "total_score": normalized_score,
        "risk_level": risk_level,
        "salary_mentioned": salary_mentioned,
        "unrealistic_salary": unrealistic_salary,
        "company_present": company_present,
    }


def validate_job_post(text):
    cleaned_text = clean_input(text)
    if cleaned_text is None:
        return {
            "company_name": "No",
            "salary_mentioned": "No",
            "unrealistic_salary": "No",
            "professional_language": "mixed",
            "contact_info": "No",
            "urgency_detected": "No",
            "personal_data_request": "No",
        }

    text_lower = cleaned_text.lower()

    company_patterns = [
        r"(?:company|employer|organization|business)\s*[:\-]?\s*([A-Z][a-zA-Z\s]+)",
        r"(?:at|for|with|at the)\s+([A-Z][a-zA-Z]{2,}(?:\s+(?:Inc|LLC|Corp|Company|Ltd))?)",
        r"\-\s*([A-Z][a-zA-Z]{2,}(?:\s+(?:Inc|LLC|Corp|Company|Ltd))?)",
        r"^([A-Z][a-zA-Z]{2,}(?:\s+(?:Inc|LLC|Corp|Company|Ltd))?)\s*[\-:]",
    ]
    company_name_found = False
    for pattern in company_patterns:
        matches = re.findall(pattern, cleaned_text, re.MULTILINE)
        for match in matches:
            if len(match.strip()) > 2 and match.strip() not in [
                "Company",
                "Confidential",
                "Unknown",
                "Major",
                "Top",
                "Leading",
                "Inc",
                "LLC",
                "Corp",
                "Ltd",
            ]:
                company_name_found = True
                break
        if company_name_found:
            break

    company_red_flags = [
        r"confidential",
        r"undisclosed",
        r"not disclosed",
        r"unnamed",
    ]
    for flag in company_red_flags:
        if re.search(r"\b" + flag + r"\b", text_lower):
            company_name_found = False
            break

    salary_patterns = [
        r"\$[\d,]+",
        r"salary[\s:]*[\$]?[\d,]+",
        r"[\d,]+(?:\s*(?:k|K))?(?:\s*(?:per|/)\s*(?:month|week|day))?",
        r"competitive pay",
        r"negotiable",
    ]
    salary_mentioned = False
    for pattern in salary_patterns:
        if re.search(pattern, text_lower):
            salary_mentioned = True
            break

    unrealistic_salary = False
    if salary_mentioned:
        high_salary_pattern = r"\$?([\d,]{5,})"
        matches = re.findall(high_salary_pattern, text_lower)
        for match in matches:
            num = safe_parse_int(match.replace(",", ""), 0)
            if num > 10000 and (
                "no experience" in text_lower or "entry level" in text_lower
            ):
                unrealistic_salary = True
                break

    professional_keywords = [
        "qualifications",
        "requirements",
        "responsibilities",
        "benefits",
        "experience",
        "skills",
        "education",
        "degree",
        "bachelor",
        "master",
        "minimum",
        "preferred",
        "competitive",
        "collaborate",
        "team",
    ]
    suspicious_keywords = [
        "guaranteed",
        "easy money",
        "quick money",
        "no interview",
        "work from home",
        "urgent",
        "immediate",
    ]

    professional_count = sum(1 for kw in professional_keywords if kw in text_lower)
    suspicious_count = sum(1 for kw in suspicious_keywords if kw in text_lower)

    if professional_count > suspicious_count and suspicious_count < 2:
        language_quality = "professional"
    elif suspicious_count > professional_count:
        language_quality = "suspicious"
    else:
        language_quality = "mixed"

    contact_patterns = [
        r"\S+@\S+\.\S+",
        r"www\.",
        r"http",
        r"phone[\s:]*[\d\-\(\)\+]+",
        r"tel[\s:]*[\d\-\(\)\+]+",
        r"apply at",
        r"visit our website",
        r"career page",
    ]
    contact_found = False
    for pattern in contact_patterns:
        if re.search(pattern, text_lower):
            contact_found = True
            break

    urgency_patterns = [
        r"urgent",
        r"immediate",
        r"apply now",
        r"limited time",
        r"last chance",
        r"act fast",
        r"hiring now",
        r"few spots",
    ]
    urgency_found = False
    for pattern in urgency_patterns:
        if re.search(pattern, text_lower):
            urgency_found = True
            break

    personal_data_patterns = [
        r"bank account",
        r"send bank",
        r"ssn",
        r"social security",
        r"credit card",
        r"gift cards",
        r"wire transfer",
        r"processing fee",
        r"registration fee",
        r"joining fee",
    ]
    personal_data_found = False
    for pattern in personal_data_patterns:
        if re.search(pattern, text_lower):
            personal_data_found = True
            break

    return {
        "company_name": "Yes" if company_name_found else "No",
        "salary_mentioned": "Yes" if salary_mentioned else "No",
        "unrealistic_salary": "Yes" if unrealistic_salary else "No",
        "professional_language": language_quality,
        "contact_info": "Yes" if contact_found else "No",
        "urgency_detected": "Yes" if urgency_found else "No",
        "personal_data_request": "Yes" if personal_data_found else "No",
    }


def calibrate_confidence(ml_probability):
    prob = safe_parse_int(ml_probability, 0)
    if prob >= 0.75:
        return "High"
    elif prob >= 0.50:
        return "Medium"
    else:
        return "Low"


def generate_explanation(
    prediction, ml_probability, risk_score_result, checklist_result
):
    pred_int = safe_parse_int(prediction, 0)
    is_fake = pred_int == 1
    risk_level = risk_score_result.get("risk_level", "low")

    reasons = []
    signals = []

    if risk_score_result.get("salary_mentioned"):
        if risk_score_result.get("unrealistic_salary"):
            reasons.append("Extremely high salary offered without requiring experience")
            signals.append("unrealistic_salary")

    if not risk_score_result.get("company_present", True):
        reasons.append("Company identity is hidden or not disclosed")
        signals.append("missing_company")

    if risk_level == "high":
        reasons.append("Multiple urgency-based language patterns detected")
        signals.append("urgency_language")
        reasons.append("Keyword pattern matches known scam templates")
        signals.append("scam_keywords")

    if checklist_result.get("professional_language") == "suspicious":
        reasons.append("Non-professional language patterns detected")
        signals.append("language_issues")

    if not checklist_result.get("contact_info"):
        reasons.append("No verifiable contact information provided")
        signals.append("missing_contact")

    if checklist_result.get("urgency_detected") == "Yes":
        reasons.append("Urgency tactics detected (immediate hiring, limited time)")
        signals.append("urgency_tactics")

    if checklist_result.get("personal_data_request") == "Yes":
        reasons.append("Request for personal/financial information detected")
        signals.append("personal_data_request")

    why_risky = ""
    if is_fake:
        why_risky = f"This job is flagged as {risk_level.upper()} RISK because: "
        if reasons:
            why_risky += "; ".join(reasons[:3]) + "."
        else:
            why_risky += "Multiple patterns match known job scams."
    else:
        if risk_level == "high":
            why_risky = "While some suspicious elements were detected, this could be a legitimate job with enthusiastic language. Additional verification recommended."
        elif risk_level == "medium":
            why_risky = "This job posting has mixed signals. Some elements are suspicious while others appear professional. Proceed with caution."
        else:
            why_risky = "This job posting shows characteristics consistent with genuine listings. The text appears professional and aligned with standard job descriptions."

    ml_prob = safe_parse_int(ml_probability, 0)
    if ml_prob >= 0.80:
        confidence_explanation = f"The ML model shows high confidence ({ml_prob * 100:.0f}% probability) based on learned patterns from training data."
    elif ml_prob >= 0.60:
        confidence_explanation = f"The ML model shows moderate confidence ({ml_prob * 100:.0f}%). Rule-based analysis provides additional validation."
    else:
        confidence_explanation = f"Lower ML confidence ({ml_prob * 100:.0f}%). Our rule-based system helps catch what the model might miss."

    model_insight = ""
    if ml_prob >= 0.7:
        model_insight = "The machine learning model strongly identifies this as a potential scam based on text patterns learned from historical fake job postings."
    elif ml_prob >= 0.4:
        model_insight = "The ML model is moderately confident. The final decision combines both ML prediction and rule-based risk scoring."
    else:
        model_insight = "The ML model indicates this may be legitimate, but rule-based checks are still performed for safety."

    decision_reasoning = []
    if checklist_result.get("personal_data_request") == "Yes":
        decision_reasoning.append("Request for personal data is a major red flag")
    if checklist_result.get("urgency_detected") == "Yes":
        decision_reasoning.append("Urgency tactics suggest potential scam")
    if checklist_result.get("company_name") == "No":
        decision_reasoning.append("Company identity not verified")
    if checklist_result.get("unrealistic_salary") == "Yes":
        decision_reasoning.append("Unrealistic salary promises detected")
    if checklist_result.get("professional_language") == "suspicious":
        decision_reasoning.append(
            "Language patterns differ from professional job postings"
        )
    if not decision_reasoning:
        decision_reasoning.append(
            "No major red flags detected through validation checks"
        )

    return {
        "why_risky": why_risky,
        "reasons": reasons[:5],
        "detected_signals": list(set(signals)),
        "confidence_explanation": confidence_explanation,
        "model_insight": model_insight,
        "decision_reasoning": decision_reasoning,
    }


def get_user_suggestions(prediction, risk_level):
    pred_int = safe_parse_int(prediction, 0)

    if pred_int == 1 or risk_level == "high":
        return [
            "Do NOT respond to this job posting or provide any information",
            "Do NOT send any money, gift cards, or processing fees",
            "Do NOT share bank account details for direct deposit setup",
            "Report this posting to the platform where you found it",
            "Search online to see if others have reported this company/job",
        ]
    elif risk_level == "medium":
        return [
            "Exercise caution when dealing with this employer",
            "Verify the company through official channels before proceeding",
            "Do not provide sensitive information until interview is completed",
            "Request a phone or video interview to verify identity",
            "Check for proper job contracts and legal compliance",
        ]
    else:
        return [
            "This appears to be a legitimate job posting",
            "Still verify the company through official channels",
            "Apply through proper channels and follow standard procedures",
            "Review the job description carefully before applying",
        ]


def calculate_hybrid_decision(ml_probability, rule_score):
    ml_prob = float(ml_probability)
    rule = safe_parse_int(rule_score, 0)
    rule_normalized = rule / 100.0
    final_score = (ml_prob * 0.6) + (rule_normalized * 0.4)

    if rule >= 70:
        prediction = "fake"
        risk_level = "high"
    elif ml_prob >= 0.6 and rule >= 40:
        prediction = "fake"
        risk_level = "high"
    elif ml_prob < 0.4 and rule < 40:
        prediction = "real"
        risk_level = "low"
    elif ml_prob < 0.4 and rule >= 40:
        prediction = "suspicious"
        risk_level = "medium"
    elif ml_prob >= 0.6 and rule < 40:
        prediction = "suspicious"
        risk_level = "medium"
    else:
        if rule >= 50:
            prediction = "fake"
            risk_level = "medium"
        elif rule >= 30:
            prediction = "suspicious"
            risk_level = "medium"
        else:
            prediction = "real"
            risk_level = "low"

    confidence = calibrate_confidence(final_score)
    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "final_score": round(final_score, 3),
        "confidence": confidence,
    }


def home(request):
    result = None

    if request.method == "POST":
        if not model_loaded:
            result = {"error": "Model not loaded. Please train the model first."}
            return render(request, "index.html", {"result": result})

        text = request.POST.get("job_description", "")

        if not text or len(text.strip()) < 10:
            result = {"error": "Please enter a job description (minimum 10 characters)"}
            return render(request, "index.html", {"result": result})

        cleaned = clean_input(text)

        if cleaned is None or len(cleaned) < 10:
            result = {
                "error": "Job description is too short. Please provide more details."
            }
            return render(request, "index.html", {"result": result})

        text_tfidf = vectorizer.transform([cleaned])

        try:
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            ml_fake_prob = float(probabilities[1])
            ml_real_prob = float(probabilities[0])
        except Exception as e:
            print(f"ML prediction failed: {e}")
            ml_fake_prob = 0.5
            ml_real_prob = 0.5
            prediction = 0

        risk_score_result = calculate_risk_score(cleaned)
        rule_score = risk_score_result["total_score"]
        rule_score_percent = (rule_score / 25.0) * 100 if rule_score > 0 else 0

        keyword_result = detect_keywords(cleaned)
        checklist_result = validate_job_post(cleaned)

        hybrid_result = calculate_hybrid_decision(ml_fake_prob, rule_score_percent)

        explanation_result = generate_explanation(
            int(prediction),
            hybrid_result["final_score"],
            risk_score_result,
            checklist_result,
        )

        suggestions = get_user_suggestions(
            1 if hybrid_result["prediction"] == "fake" else 0,
            hybrid_result["risk_level"],
        )

        reasons = []
        if risk_score_result.get("unrealistic_salary"):
            reasons.append("Unrealistic high salary with no experience required")
        if not risk_score_result.get("company_present", True):
            reasons.append("Company identity is hidden or confidential")
        if rule_score >= 10:
            reasons.append("Multiple suspicious keywords detected")
        if rule_score >= 5:
            reasons.append("Several red flag phrases found in text")
        if checklist_result.get("professional_language") == "suspicious":
            reasons.append("Non-professional language patterns detected")
        if not checklist_result.get("contact_info"):
            reasons.append("No verifiable contact information provided")
        if ml_fake_prob >= 0.7:
            reasons.append(
                "ML model strongly detects this as fake based on text patterns"
            )

        all_keywords = []
        for category, keywords in keyword_result.get("categories", {}).items():
            all_keywords.extend(keywords)
        all_keywords = sorted(all_keywords, key=lambda x: x["weight"], reverse=True)[:8]

        result = {
            "prediction": hybrid_result["prediction"],
            "risk_level": hybrid_result["risk_level"],
            "confidence": hybrid_result["confidence"],
            "ml_probability": round(ml_fake_prob * 100, 1),
            "rule_score": round(rule_score_percent, 0),
            "final_score": round(ml_fake_prob * 100, 1),
            "real_probability": round(ml_real_prob * 100, 1),
            "fake_probability": round(ml_fake_prob * 100, 1),
            "risk_score": rule_score,
            "risk_breakdown": {
                "keyword_score": risk_score_result.get("keyword_score", 0),
                "salary_score": risk_score_result.get("salary_score", 0),
                "company_score": risk_score_result.get("company_score", 0),
            },
            "reasons": reasons if reasons else ["No significant risk factors detected"],
            "suspicious_keywords": [
                {"keyword": k["keyword"], "weight": k["weight"]} for k in all_keywords
            ],
            "checklist": checklist_result,
            "suggestions": suggestions,
            "explanation": explanation_result,
            "input_text": text,
        }

    return render(request, "index.html", {"result": result})
