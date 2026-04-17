"""
Gemini AI client for AMDGT web app.
Provides natural-language explanations for drug-disease predictions,
graph summaries, and molecule candidate rationale.
"""
import os
import json

try:
    from google import genai
    from google.genai import types as genai_types
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

_client = None
_api_key = None
_MODEL = "gemini-2.0-flash"


def configure(api_key: str):
    global _client, _api_key
    if not _SDK_AVAILABLE:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")
    _api_key = api_key
    _client = genai.Client(api_key=api_key)


def is_ready() -> bool:
    return _SDK_AVAILABLE and _client is not None


def _ask(prompt: str, max_tokens: int = 600) -> str:
    if not is_ready():
        raise RuntimeError("Gemini chưa được cấu hình. Vui lòng thêm API key.")
    resp = _client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.4,
        ),
    )
    return resp.text.strip()


# ── Public functions ───────────────────────────────────────────

def explain_prediction(drug_name: str, disease_name: str, score: float,
                        is_known: bool, fuzzy_details: dict | None = None) -> str:
    """Explain a single drug-disease prediction in Vietnamese."""
    known_str = "đã được xác nhận trong dữ liệu thực nghiệm" if is_known else "được dự đoán bởi mô hình AI"
    score_level = (
        "rất cao (≥0.7)" if score >= 0.7 else
        "cao (0.5–0.7)"  if score >= 0.5 else
        "trung bình (0.3–0.5)" if score >= 0.3 else
        "thấp (<0.3)"
    )
    fuzzy_str = ""
    if fuzzy_details:
        cf  = fuzzy_details.get("cf_score", "N/A")
        src = fuzzy_details.get("src_neighbor", "N/A")
        tgt = fuzzy_details.get("tgt_neighbor", "N/A")
        fuzzy_str = f"""
Các điểm đầu vào Fuzzy:
- CF Score (lọc cộng tác): {cf:.3f}
- Src Neighbor (tương đồng láng giềng nguồn): {src:.3f}
- Tgt Neighbor (tương đồng láng giềng đích): {tgt:.3f}"""

    prompt = f"""Bạn là chuyên gia dược lý và tin sinh học. Hãy giải thích ngắn gọn (3-5 câu, bằng tiếng Việt) về mối liên kết dự đoán sau:

Thuốc: {drug_name}
Bệnh: {disease_name}
Điểm liên kết: {score:.4f} ({score_level})
Trạng thái: {known_str}{fuzzy_str}

Hãy:
1. Giải thích ý nghĩa điểm số này
2. Nêu cơ chế sinh học tiềm năng nếu có
3. Đưa ra nhận xét về độ tin cậy của dự đoán
Trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu."""
    return _ask(prompt, max_tokens=400)


def explain_matrix(drugs: list[str], diseases: list[str],
                   matrix: list[list[dict]], model_name: str) -> str:
    """Summarise a drug×disease matrix result in Vietnamese."""
    # Build a compact text summary of the matrix
    top_pairs = []
    for di, drug in enumerate(drugs):
        for dj, disease in enumerate(diseases):
            cell = matrix[di][dj]
            top_pairs.append((drug, disease, cell["score"], cell["is_known"]))
    top_pairs.sort(key=lambda x: -x[2])
    top5 = top_pairs[:5]
    table = "\n".join(
        f"  - {d} → {dis}: {sc:.3f} {'[đã biết]' if k else '[dự đoán]'}"
        for d, dis, sc, k in top5
    )
    prompt = f"""Bạn là chuyên gia dược lý. Dưới đây là ma trận dự đoán liên kết Thuốc–Bệnh từ mô hình {model_name}.

Thuốc được phân tích: {', '.join(drugs)}
Bệnh được phân tích: {', '.join(diseases)}

Top 5 cặp có điểm cao nhất:
{table}

Hãy viết một đoạn phân tích tổng hợp (4-6 câu, tiếng Việt):
1. Cặp thuốc-bệnh nào tiềm năng nhất?
2. Có cặp nào bất ngờ hoặc đáng chú ý?
3. Khuyến nghị gì cho nghiên cứu tiếp theo?"""
    return _ask(prompt, max_tokens=500)


def explain_graph(entity_type: str, entity_name: str,
                  neighbors: list[dict], model_name: str) -> str:
    """Explain a network graph's neighbours in Vietnamese."""
    top = sorted(neighbors, key=lambda x: -x.get("score", 0))[:6]
    lines = "\n".join(
        f"  - {n.get('label','?')}: {n.get('score',0):.3f} {'[đã biết]' if n.get('is_known') else ''}"
        for n in top
    )
    entity_vn = "thuốc" if entity_type == "drug" else "bệnh"
    prompt = f"""Bạn là chuyên gia tin sinh học. Phân tích đồ thị mạng lưới:

Thực thể trung tâm: {entity_name} (loại: {entity_vn})
Mô hình: {model_name}
Các liên kết hàng đầu:
{lines}

Hãy phân tích (3-5 câu, tiếng Việt):
1. Ý nghĩa của các liên kết quan trọng nhất
2. Có pattern nào thú vị trong mạng lưới?
3. Gợi ý hướng nghiên cứu."""
    return _ask(prompt, max_tokens=400)


def explain_molecule(disease_name: str, candidates: list[dict]) -> str:
    """Explain generated drug candidate molecules in Vietnamese, validating disease fit."""
    mols = "\n".join(
        f"  {i+1}. {c.get('name','?')} — Điểm Fuzzy: {c.get('score',0):.3f} | Chiến lược: {c.get('strategy','?')} | SMILES: {c.get('smiles','?')[:50]}"
        for i, c in enumerate(candidates[:5])
    )
    prompt = f"""Bạn là chuyên gia hóa dược và dược lý học. Hệ thống AI đã tổng hợp các phân tử thuốc ứng viên nhắm vào bệnh: **{disease_name}**

Danh sách ứng viên phân tử:
{mols}

Hãy phân tích và đánh giá (5-7 câu, bằng tiếng Việt):
1. **Tính phù hợp bệnh lý**: Các phân tử này có phù hợp với cơ chế bệnh sinh của "{disease_name}" không? Giải thích ngắn gọn.
2. **Ứng viên tiềm năng nhất**: Phân tử nào có điểm Fuzzy cao nhất và cấu trúc hóa học phù hợp nhất — tại sao?
3. **Đánh giá chiến lược tổng hợp**: Fragment Addition vs Scaffold Hybridization — phương pháp nào hiệu quả hơn cho bệnh này?
4. **Cảnh báo**: Những rủi ro nào về tính độc hại, khả năng hấp thu (ADME), hoặc phản ứng chéo cần lưu ý?
5. **Bước tiếp theo**: Khuyến nghị hướng nghiên cứu thực nghiệm (in vitro / in vivo) cho ứng viên tốt nhất.
Trả lời bằng tiếng Việt, súc tích và có tính chuyên môn cao."""
    return _ask(prompt, max_tokens=600)


def explain_fuzzy_layer(drug_name: str, disease_name: str,
                        score: float, n_rules: int,
                        top_rules: list[dict]) -> str:
    """
    Explain how the GNN+Fuzzy model processed a drug-disease pair.

    Parameters
    ----------
    top_rules : list of dicts with keys 'rule_id' (int) and 'strength' (float),
                sorted descending by strength.
    """
    rules_str = "\n".join(
        f"  Luật #{r['rule_id']+1}: cường độ kích hoạt = {r['strength']:.4f}"
        for r in top_rules[:6]
    )
    score_level = (
        "rất cao (≥0.7)" if score >= 0.7 else
        "cao (0.5–0.7)"  if score >= 0.5 else
        "trung bình (0.3–0.5)" if score >= 0.3 else
        "thấp (<0.3)"
    )
    prompt = f"""Bạn là chuyên gia AI y sinh. Mô hình GNN+Fuzzy (AMNTDDA với Learnable Fuzzy Layer gồm {n_rules} luật mờ TSK) vừa dự đoán liên kết:

Thuốc: {drug_name}
Bệnh: {disease_name}
Điểm liên kết: {score:.4f} ({score_level})

Các luật mờ được kích hoạt mạnh nhất:
{rules_str}

Giải thích (4-6 câu, tiếng Việt):
1. Ý nghĩa của các luật mờ được kích hoạt mạnh – chúng nắm bắt đặc trưng gì trong không gian tương tác thuốc–bệnh?
2. Tại sao mô hình lai GNN+Fuzzy có thể xử lý nhiễu và sự không chắc chắn tốt hơn GNN thuần túy?
3. Điểm liên kết {score:.4f} có độ tin cậy cao không xét theo phân tích luật mờ này?"""
    return _ask(prompt, max_tokens=450)


def explain_fuzzy_animation(n_rules: int, drug_name: str = '', disease_name: str = '') -> str:
    """
    Generate a short Vietnamese description of what the fuzzy animation is showing,
    suitable for a tooltip or side-panel on the web page.
    """
    context = f"đang xử lý cặp thuốc '{drug_name}' và bệnh '{disease_name}'" if drug_name else ""
    prompt = f"""Bạn là chuyên gia AI. Mô tả ngắn gọn (3 câu, tiếng Việt, không dùng markdown) về animation Fuzzy Neural Network sau đây:

Animation hiển thị {n_rules} hàm thành viên Gaussian (Gaussian Membership Functions) của lớp Fuzzy học được (Learnable Fuzzy Layer) {context}.
Màu sắc biểu thị mức độ kích hoạt của mỗi luật mờ. Nhiễu Perlin nền thể hiện sự không chắc chắn vốn có của dự đoán.

Hãy giải thích đơn giản để người dùng hiểu animation này đang mô phỏng điều gì."""
    return _ask(prompt, max_tokens=200)
