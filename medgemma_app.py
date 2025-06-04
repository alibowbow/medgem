import streamlit as st
import requests
import base64
from PIL import Image
import io

# 페이지 설정
st.set_page_config(
    page_title="MedGemma 의료 이미지 분석",
    page_icon="🏥",
    layout="wide"
)

# 제목
st.title("🏥 MedGemma 의료 이미지 분석 시스템")
st.markdown("Hugging Face의 MedGemma 모델을 활용한 X-ray 이미지 및 임상정보 분석")

# 사이드바 - 모드 선택
st.sidebar.header("🧠 분석 모드 선택")
mode = st.sidebar.selectbox(
    "원하시는 분석 모드를 선택하세요:",
    ["이미지 설명", "질문 응답", "리포트 생성", "이미지 비교", "임상추론 (텍스트)"]
)

# API 키 입력
st.sidebar.header("🔐 API 설정")
api_key = st.sidebar.text_input(
    "Hugging Face API 키를 입력하세요:",
    type="password",
    help="https://huggingface.co/settings/tokens 에서 발급받을 수 있습니다."
)

if api_key:
    st.sidebar.success("✅ API 키가 입력되었습니다.")
    
# 모델 상태 확인
st.sidebar.header("📊 모델 상태")
if api_key:
    if st.sidebar.button("모델 상태 확인"):
        with st.spinner("모델 상태 확인 중..."):
            # MedGemma-4b-it 상태 확인
            headers = {"Authorization": f"Bearer {api_key}"}
            try:
                response = requests.get(
                    "https://api-inference.huggingface.co/models/google/medgemma-4b-it",
                    headers=headers
                )
                if response.status_code == 200:
                    st.sidebar.success("MedGemma-4b-it: ✅ 사용 가능")
                else:
                    st.sidebar.warning("MedGemma-4b-it: ⏳ 준비 중")
            except:
                st.sidebar.error("MedGemma-4b-it: ❌ 확인 실패")

# 이미지를 base64로 변환하는 함수
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# MedGemma API 호출 함수
def call_medgemma_vision(prompt, image_b64, api_key, model="google/medgemma-4b-it"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": [               # 반드시 리스트!
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image_b64}
                ]
            }
        ]
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.ok:
        data = r.json()
        # MedGemma는 마지막 토큰에 답변이 들어갑니다
        try:
            return data[0]["generated_text"][-1]["content"]
        except Exception:
            return data
    elif r.status_code == 503:
        return {"error": "⏳ 모델이 로딩 중입니다. 잠시 후 다시 시도해 주세요."}
    else:
        return {"error": f"API 오류: {r.status_code} - {r.text}"}

# MedGemma 텍스트 전용 API 호출 함수
def call_medgemma_text(prompt, api_key, model="google/medgemma-27b"):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
        "parameters": {"max_new_tokens": 400, "temperature": 0.7}
    }
    
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.ok:
        return r.json()
    else:
        return f"API 오류: {r.status_code} - {r.text}"

# 메인 컨텐츠 영역
if not api_key:
    st.warning("⚠️ 사이드바에서 Hugging Face API 키를 입력해주세요.")
else:
    # 모드별 처리
    if mode == "이미지 설명":
        st.header("📸 이미지 설명 모드")
        st.markdown("X-ray 이미지를 업로드하면 AI가 자동으로 설명을 생성합니다.")
        
        uploaded_file = st.file_uploader("X-ray 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("업로드된 이미지")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("AI 분석 결과")
                if st.button("이미지 분석 시작"):
                    with st.spinner("이미지를 분석 중입니다..."):
                        image_base64 = image_to_base64(image)
                        prompt = "Please describe this medical image in detail."
                        result = call_medgemma_vision(prompt, image_base64, api_key)
                        
                        if isinstance(result, dict) and "error" in result:
                            st.error(result["error"])
                        else:
                            st.success("분석 완료!")
                            st.write(result)
    
    elif mode == "질문 응답":
        st.header("❓ 질문 응답 모드")
        st.markdown("X-ray 이미지와 함께 질문을 입력하면 AI가 답변합니다.")
        
        uploaded_file = st.file_uploader("X-ray 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
        question = st.text_area("이미지에 대한 질문을 입력하세요:", height=100)
        
        if uploaded_file is not None and question:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("업로드된 이미지")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("AI 답변")
                if st.button("질문에 답변받기"):
                    with st.spinner("답변을 생성 중입니다..."):
                        image_base64 = image_to_base64(image)
                        result = call_medgemma_vision(question, image_base64, api_key)
                        
                        if isinstance(result, dict) and "error" in result:
                            st.error(result["error"])
                        else:
                            st.success("답변 생성 완료!")
                            st.write(result)
    
    elif mode == "리포트 생성":
        st.header("📄 리포트 생성 모드")
        st.markdown("X-ray 이미지를 업로드하면 방사선과 스타일의 상세 리포트를 생성합니다.")
        
        uploaded_file = st.file_uploader("X-ray 이미지를 업로드하세요", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("업로드된 이미지")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("방사선과 리포트")
                if st.button("리포트 생성"):
                    with st.spinner("리포트를 생성 중입니다..."):
                        image_base64 = image_to_base64(image)
                        prompt = """Generate a detailed radiology report for this medical image. 
                        Include sections for: Technique, Findings, Impression, and Recommendations.
                        Use professional radiology terminology."""
                        result = call_medgemma_vision(prompt, image_base64, api_key)
                        
                        if isinstance(result, dict) and "error" in result:
                            st.error(result["error"])
                        else:
                            st.success("리포트 생성 완료!")
                            st.write(result)
    
    elif mode == "이미지 비교":
        st.header("🔄 이미지 비교 모드")
        st.markdown("두 개의 X-ray 이미지를 업로드하면 AI가 변화를 분석합니다.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("이전 이미지")
            uploaded_file1 = st.file_uploader("첫 번째 이미지 업로드", type=['png', 'jpg', 'jpeg'], key="img1")
            if uploaded_file1:
                image1 = Image.open(uploaded_file1)
                st.image(image1, use_column_width=True)
        
        with col2:
            st.subheader("최근 이미지")
            uploaded_file2 = st.file_uploader("두 번째 이미지 업로드", type=['png', 'jpg', 'jpeg'], key="img2")
            if uploaded_file2:
                image2 = Image.open(uploaded_file2)
                st.image(image2, use_column_width=True)
        
        if uploaded_file1 and uploaded_file2:
            st.subheader("비교 분석 결과")
            if st.button("이미지 비교 분석"):
                with st.spinner("이미지를 비교 분석 중입니다..."):
                    # 두 이미지를 하나로 합치기
                    combined_width = image1.width + image2.width
                    combined_height = max(image1.height, image2.height)
                    combined_image = Image.new('RGB', (combined_width, combined_height))
                    combined_image.paste(image1, (0, 0))
                    combined_image.paste(image2, (image1.width, 0))
                    
                    combined_base64 = image_to_base64(combined_image)
                    prompt = """Compare these two medical images (left: earlier, right: later).
                    Analyze and describe any changes, improvements, deterioration, or stability.
                    Provide specific observations about the differences."""
                    
                    result = call_medgemma_vision(prompt, combined_base64, api_key)
                    
                    if isinstance(result, dict) and "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("비교 분석 완료!")
                        st.write(result)
    
    elif mode == "임상추론 (텍스트)":
        st.header("🧬 임상추론 모드")
        st.markdown("임상 정보를 입력하면 MedGemma-27B 모델이 의학적 추론을 제공합니다.")
        
        clinical_text = st.text_area(
            "임상 정보를 입력하세요:",
            height=200,
            placeholder="예: 45세 남성 환자, 3일 전부터 시작된 흉통, 운동 시 악화..."
        )
        
        if clinical_text:
            if st.button("임상 추론 시작"):
                with st.spinner("임상 추론을 수행 중입니다..."):
                    prompt = f"""Based on the following clinical information, provide a comprehensive clinical reasoning:
                    
                    {clinical_text}
                    
                    Please include:
                    1. Differential diagnosis
                    2. Recommended investigations
                    3. Initial management approach
                    4. Key considerations"""
                    
                    result = call_medgemma_text(prompt, api_key)
                    
                    st.subheader("AI 임상 추론 결과")
                    if isinstance(result, str) and "API 오류" in result:
                        st.error(result)
                    else:
                        st.success("임상 추론 완료!")
                        # 결과 처리
                        if isinstance(result, list) and len(result) > 0:
                            # 대화형 응답에서 마지막 메시지 추출
                            try:
                                content = result[0]["generated_text"][-1]["content"]
                                st.write(content)
                            except:
                                st.write(result)
                        else:
                            st.write(result)

# 하단 정보
st.markdown("---")
st.markdown("""
**⚠️ 주의사항:**
- 이 시스템은 의료 진단 보조 도구이며, 최종 진단은 반드시 전문 의료진이 수행해야 합니다.
- 업로드된 이미지는 분석 후 즉시 삭제됩니다.
- API 키는 안전하게 보관하시고 타인과 공유하지 마세요.

**💡 문제 해결:**
- API 오류 404: 모델이 아직 로딩 중일 수 있습니다. 사이드바에서 모델 상태를 확인해주세요.
- API 오류 503: 서버가 과부하 상태입니다. 잠시 후 다시 시도해주세요.
- 결과가 나오지 않는 경우: API 키가 올바른지 확인하고, 모델 상태를 확인해주세요.
""")

# 모델 정보
with st.expander("📚 MedGemma 모델 정보"):
    st.markdown("""
    **MedGemma-4b-it**: 
    - 의료 이미지 분석에 특화된 4B 파라미터 모델
    - Vision-Language 태스크 수행 가능
    - X-ray, CT, MRI 등 다양한 의료 이미지 분석
    
    **MedGemma-27B**: 
    - 대규모 언어 모델 기반의 임상 추론 특화
    - 의학 문헌과 임상 데이터로 학습
    - 복잡한 의료 케이스 분석 가능
    
    **참고**: 이 모델들은 Google Research와 DeepMind가 개발한 의료 AI 모델입니다.
    """)

# 사용 가이드
with st.expander("📖 사용 가이드"):
    st.markdown("""
    1. **API 키 발급**: 
       - https://huggingface.co/settings/tokens 접속
       - 'New token' 클릭하여 토큰 생성
       - 'read' 권한만 있으면 충분
    
    2. **이미지 준비**:
       - PNG, JPG, JPEG 형식 지원
       - 선명한 의료 이미지 권장
       - 파일 크기 10MB 이하 권장
    
    3. **모드별 팁**:
       - **이미지 설명**: 전반적인 소견을 빠르게 확인할 때
       - **질문 응답**: 특정 부위나 증상에 대해 질문할 때
       - **리포트 생성**: 공식적인 문서가 필요할 때
       - **이미지 비교**: 치료 전후 비교나 경과 관찰 시
       - **임상추론**: 증상과 병력만으로 추론이 필요할 때
    """)
