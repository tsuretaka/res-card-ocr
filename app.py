import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google.cloud import vision
import io
from PIL import Image
import json
import re
import os
import cv2
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="予約カードOCRシステム",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; padding: 0.5rem 2rem; border-radius: 50px;
        font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    h1 {
        background: -webkit-linear-gradient(45deg, #1a1a1a, #4a4a4a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stFileUploader"] section > div > div > span,
    [data-testid="stFileUploader"] section > div > div > small {
        display: none !important;
    }
    
    [data-testid="stFileUploader"] section > div > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px; 
    }

    [data-testid="stFileUploader"] section > div > div::after {
        content: "ここをタップしてカメラ起動または画像選択";
        display: block;
        order: -1; 
        color: #555;
        font-weight: bold;
        margin-top: -10px; 
        margin-bottom: 5px;
    }
    
    [data-testid="stFileUploader"] button {
        color: transparent !important;
        min-width: 200px; 
        min-height: 50px; 
        position: relative !important;
        border: 1px solid rgba(0,0,0,0.1); 
        border-radius: 8px;
    }

    [data-testid="stFileUploader"] button::before {
        content: "📸 カメラ / 📁 アルバム";
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #333 !important; 
        font-size: 1.1rem;
        font-weight: bold;
        pointer-events: none; 
    }
    
    [data-testid="stFileUploader"] button:hover {
        border-color: #4facfe;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 2rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        h1 {
            font-size: 1.5rem !important;
            line-height: 1.4 !important;
            text-align: left; 
            margin-bottom: 1.5rem !important;
            display: flex;
            align-items: center;
        }
        .stButton>button {
            width: 100%;
            padding: 0.8rem 1rem;
            font-size: 1rem;
            margin-bottom: 10px;
        }
        input, textarea {
            font-size: 16px !important; 
        }
        [data-testid="stSidebar"] {
            width: 100% !important;
        }
    }
    
    .footer {
        width: 100%;
        text-align: center;
        padding: 3rem 0 1rem 0;
        margin-top: 2rem;
        color: #888;
        font-size: 0.85rem;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1McrtrFeMCufGrzVJgaKFGJMyO5kSLnv9hEHGnah9t4A/edit?usp=sharing"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/cloud-vision"]

def load_credentials(source):
    try:
        if isinstance(source, str):
            creds = Credentials.from_service_account_file(source, scopes=SCOPES)
        elif isinstance(source, dict):
            creds = Credentials.from_service_account_info(source, scopes=SCOPES)
        else:
            creds_dict = json.load(source)
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        return creds
    except Exception as e:
        st.error(f"認証エラー: {e}")
        return None

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    _, encoded_img = cv2.imencode('.jpg', enhanced)
    return encoded_img.tobytes(), enhanced

def perform_ocr(image_content, credentials):
    try:
        client = vision.ImageAnnotatorClient(credentials=credentials)
        image = vision.Image(content=image_content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if response.error.message:
            st.error(f"OCR Error: {response.error.message}")
            return None
        return texts[0].description if texts else ""
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def parse_ocr_residue(text):
    data = {
        "氏名": "", "年齢": "", "職業": "", "住所": "",
        "電話番号": "", "メールアドレス": "", "チェックイン日": "", "チェックアウト日": ""
    }
    
    full_text = text.replace('\n', ' ').replace('　', ' ')
    
    headers = [
        "氏名", "名前", "Name", "Guest Name", 
        "年齢", "Age", 
        "ご職業", "職業", "Occupation", "Job",
        "住所", "Address", "住 所",
        "電話番号", "電話", "Tel", "Phone", "Mobile", "Cell",
        "メールアドレス", "メール", "Email", "E-mail", "Mail",
        "チェックイン日", "チェックイン", "Check-in",
        "チェックアウト日", "チェックアウト", "Check-out",
        "お客様記入欄", "ホテル使用欄", "区分", "金額", "小計", "合計"
    ]
    residue_text = full_text
    for h in headers:
        residue_text = residue_text.replace(h, " ")

    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', full_text)
    if email_match:
        data["メールアドレス"] = email_match.group(0)
        residue_text = residue_text.replace(data["メールアドレス"], " ")

    date_matches = re.findall(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', full_text)
    found_dates = []
    for d in date_matches:
        d_str = f"{d[0]}/{d[1]}/{d[2]}"
        found_dates.append(d_str)
        pat = f"{d[0]}[./-]{d[1]}[./-]{d[2]}"
        residue_text = re.sub(pat, " ", residue_text)

    if len(found_dates) >= 2:
        found_dates.sort()
        data["チェックイン日"] = found_dates[0]
        data["チェックアウト日"] = found_dates[-1]
    elif len(found_dates) == 1:
        data["チェックイン日"] = found_dates[0]

    def normalize_num(s):
        trans = str.maketrans('０１２３４５６７８９', '0123456789')
        s = s.translate(trans)
        s = s.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1')
        return s

    pref_pattern = r'(北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|東京都|神奈川県|新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|大阪府|兵庫県|奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|福岡県|佐賀県|長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県)'
    addr_match = re.search(pref_pattern, residue_text)
    potential_phone_text = residue_text

    if addr_match:
        start_idx = addr_match.start()
        after_addr_start = residue_text[start_idx:]
        split_pattern = r'(0[789]0|Tel|Phone|Mobile)'
        split_match = re.search(split_pattern, after_addr_start, re.IGNORECASE)
        addr_end_idx = len(after_addr_start)
        if split_match:
            addr_end_idx = split_match.start()
            clean_addr = after_addr_start[:addr_end_idx].strip()
            data["住所"] = re.sub(r'[\s-]*$', '', clean_addr)
            potential_phone_text = after_addr_start[addr_end_idx:]
            residue_text = residue_text[:start_idx] + after_addr_start[addr_end_idx:]
        else:
            data["住所"] = after_addr_start.strip()
            residue_text = residue_text[:start_idx]

    norm_phone_text = normalize_num(potential_phone_text)
    p_matches = re.findall(r'(0\d[\d\s-]{8,}\d)', norm_phone_text)
    valid_phone = ""
    for p in p_matches:
        digits = re.sub(r'\D', '', p)
        if 10 <= len(digits) <= 11 and digits.startswith('0'):
             valid_phone = p
             if digits.startswith(('090', '080', '070')):
                 break
    if valid_phone:
        data["電話番号"] = valid_phone
        if data["住所"] and valid_phone in str(data["住所"]):
             data["住所"] = data["住所"].replace(valid_phone, "").strip()
        if valid_phone in residue_text:
            residue_text = residue_text.replace(valid_phone, " ")

    tokens = [t for t in residue_text.split() if t.strip()]
    final_tokens = []
    for t in tokens:
        if re.match(r'^\d{1,3}$', t):
            if not data["年齢"]:
                data["年齢"] = t
            continue
        if len(t) == 1 and not t.isalnum():
            continue
        final_tokens.append(t)
        
    if len(final_tokens) > 0:
        name_val = final_tokens[0]
        if len(final_tokens) > 1:
            second = final_tokens[1]
            job_keywords = ["会社", "代表", "役員", "社員", "教員", "公務員", "医師", "弁護士", "自営", "フリー", "主婦", "学生", "無職", "CEO", "Manager", "Director"]
            if any(k in second for k in job_keywords):
                data["職業"] = second
            else:
                name_val += " " + second
                if len(final_tokens) > 2:
                    data["職業"] = final_tokens[2]
        data["氏名"] = name_val

    return data

def validate_document_type(text):
    keywords = [
        "氏名", "名前", "Name", "Guest",
        "住所", "Address", "住 所",
        "電話", "Tel", "Phone", "Mobile",
        "チェックイン", "Check-in",
        "チェックアウト", "Check-out",
        "メール", "Email", "E-mail",
        "宿泊", "Stay", "泊",
        "署名", "Signature", "Sign",
        "Age", "年齢"
    ]
    count = 0
    for kw in keywords:
        if kw in text:
            count += 1
    return count >= 2

def main():
    local_css()
    st.title("📋 予約カードOCR転記システム")
    
    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = 0

    creds = None
    SERVICE_ACCOUNT_FILE = "service_account.json"
    
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = load_credentials(SERVICE_ACCOUNT_FILE)
        st.sidebar.success("🔑 認証キー読込済み (Local)")
    elif 'gcp_service_account' in st.secrets:
        try:
            creds_dict = dict(st.secrets['gcp_service_account'])
            creds = load_credentials(creds_dict)
            st.sidebar.success("🔑 認証キー読込済み (Secrets)")
        except Exception as e:
            st.sidebar.error(f"Secrets読込エラー: {e}")

    if not creds:
        st.sidebar.header("設定")
        creds_file = st.sidebar.file_uploader("サービスアカウントキー (JSON)", type="json")
        if creds_file:
            creds = load_credentials(creds_file)
        else:
            st.warning("⚠️ 認証キーが見つかりません。'service_account.json'を配置するか、Secretsを設定してください。")
            return

    if st.sidebar.button("🔄 リセット / 次の画像を読み込む", type="primary"):
        st.session_state['uploader_key'] += 1
        st.session_state.pop('ocr_result', None)
        st.session_state.pop('raw_text', None)
        st.session_state.pop('camera_image', None)
        st.rerun()

    # シングルボタン構成
    uploaded_image = st.file_uploader(
        "予約カードを撮影または選択", 
        type=['png', 'jpg', 'jpeg'], 
        key=f"uploader_{st.session_state['uploader_key']}",
        label_visibility="collapsed"
    )
    
    image_content = None
    final_image = None 

    if uploaded_image:
        image_content = uploaded_image.getvalue()
        final_image = Image.open(uploaded_image)

    col1, col2 = st.columns([1, 1.2]) 
    
    with col1:
        st.subheader("1. 予約カード読込")
        
        use_enhance = st.checkbox("手書き文字補正を行う (推奨)", value=True, help="文字を濃くし、影を除去して読み取りやすくします。")
        
        if final_image:
            st.image(final_image, caption='読込画像', use_container_width=True)
            
            if st.button("🔍 OCR解析実行", type="primary"):
                with st.spinner('画像補正 & 解析中...'):
                    img_byte_arr = io.BytesIO()
                    final_image.save(img_byte_arr, format=final_image.format or 'JPEG')
                    target_bytes = img_byte_arr.getvalue()
                    
                    if use_enhance:
                        target_bytes, processed_cv2_img = preprocess_image(target_bytes)
                        with st.expander("補正後の画像を確認"):
                            st.image(processed_cv2_img, caption="AIが見ている画像", clamp=True, channels='GRAY', use_container_width=True)
                    
                    full_text = perform_ocr(target_bytes, creds)
                    
                    if full_text:
                        if not validate_document_type(full_text):
                            st.warning("⚠️ 【警告】 読取エラーの可能性があります。予約カードではない、またはレイアウトが大きく異なる書類のようです。")
                            
                        parsed = parse_ocr_residue(full_text)
                        st.session_state['ocr_result'] = parsed
                        st.session_state['raw_text'] = full_text
                        st.success("解析完了")
                    else:
                        st.error("読み取り失敗")
        else:
            st.info("👆 上のボタンから「写真を撮る」または「ライブラリから選択」してください")

    with col2:
        if 'ocr_result' in st.session_state:
            st.subheader("2. データ確認・編集")
            st.info("✏️ 各項目をタップして修正できます。間違いがないかご確認ください。", icon="👆")
            data = st.session_state['ocr_result']
            
            with st.form("verify_form"):
                c1, c2 = st.columns(2)
                with c1:
                    name = st.text_input("氏名 (A列)", value=data.get("氏名"))
                    age = st.text_input("年齢 (B列)", value=data.get("年齢"))
                    job = st.text_input("ご職業 (C列)", value=data.get("職業"))
                    phone = st.text_input("電話番号 (E列)", value=data.get("電話番号"))
                with c2:
                    checkin = st.text_input("チェックイン日 (G列)", value=data.get("チェックイン日"))
                    checkout = st.text_input("チェックアウト日 (H列)", value=data.get("チェックアウト日"))
                    email = st.text_input("メールアドレス (F列)", value=data.get("メールアドレス"))
                
                address = st.text_area("住所 (D列)", value=data.get("住所"), height=100)
                
                with st.expander("OCR生データを表示"):
                    st.text_area("解析前のテキスト", st.session_state.get('raw_text', ''), height=150)

                st.markdown("---")
                submitted = st.form_submit_button("✅ 承認してスプレッドシートへ転記")
                if submitted:
                    try:
                        gc = gspread.authorize(creds)
                        sh = gc.open_by_url(SPREADSHEET_URL)
                        
                        # 1. メインシートへの転記
                        ws = sh.get_worksheet(0)
                        row = [name, age, job, address, phone, email, checkin, checkout]
                        ws.append_row(row)
                        
                        # 2. 生データのバックアップ (OCR_LOGシート)
                        try:
                            log_ws = sh.worksheet('OCR_LOG')
                        except:
                            log_ws = sh.add_worksheet(title='OCR_LOG', rows=1000, cols=50)
                            header = ['タイムスタンプ'] + [f'Line {i+1}' for i in range(49)]
                            log_ws.append_row(header)
                            
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # OCR生テキストを行ごとに分割してB列以降に展開
                        raw_text_full = st.session_state.get('raw_text', '')
                        raw_lines = [line.strip() for line in raw_text_full.splitlines() if line.strip()]
                        
                        log_row = [timestamp] + raw_lines
                        
                        # updateで確実書き込み
                        next_row = len(log_ws.col_values(1)) + 1
                        log_ws.update(range_name=f'A{next_row}', values=[log_row])
                        
                        st.success("✅ 転記完了！（生データログも保存しました）")
                        st.balloons()
                    except Exception as e:
                        st.error(f"書込エラー: {e}")

    st.markdown('<div class="footer">Developed by Center of Okinawa Local Tourism</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
