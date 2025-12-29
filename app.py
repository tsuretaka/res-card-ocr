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
import base64
import random

# FORCE DEPLOY v3.1 - Parsing Logic: Prioritize Below Header

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
    
    @keyframes floatUp {
        0% { bottom: -150px; transform: translateX(0) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        25% { transform: translateX(20px) rotate(5deg); }
        50% { transform: translateX(-20px) rotate(-5deg); }
        75% { transform: translateX(10px) rotate(3deg); opacity: 0.9; }
        100% { bottom: 100vh; transform: translateX(0) rotate(0deg); opacity: 0; }
    }
    
    .floating-container {
        position: fixed;
        left: 0; top: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 9999; overflow: hidden;
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
    
    full_text = text
    
    # 1. メールアドレス (Global search)
    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', full_text)
    if email_match:
        data["メールアドレス"] = email_match.group(0)
    
    # 2. 日付 (Global search)
    date_matches = re.findall(r'(\d{4})[\./\-](\d{1,2})[\./\-](\d{1,2})', full_text)
    found_dates = []
    for d in date_matches:
        d_str = f"{d[0]}/{d[1]}/{d[2]}"
        found_dates.append(d_str)
    
    if len(found_dates) >= 2:
        found_dates.sort()
        data["チェックイン日"] = found_dates[0]
        data["チェックアウト日"] = found_dates[-1]
    elif len(found_dates) == 1:
        data["チェックイン日"] = found_dates[0]

    # 3. 電話番号 (Global search)
    trans = str.maketrans('０１２３４５６７８９', '0123456789')
    norm_text = full_text.translate(trans)
    
    phone_pattern = r'(0\d{1,4}[\s-]?\d{1,4}[\s-]?\d{3,4})'
    p_matches = re.findall(phone_pattern, norm_text)
    
    valid_phone = ""
    for p in p_matches:
        digits = re.sub(r'\D', '', p)
        if 10 <= len(digits) <= 11 and digits.startswith('0'):
            if digits.startswith(('090', '080', '070', '03', '06', '092', '098')): 
                valid_phone = p
                break
            elif len(digits) == 10 and digits.startswith('0'):
                 valid_phone = p
    
    if valid_phone:
        data["電話番号"] = valid_phone

    # 4. 行ごとの解析
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    pref_pattern = r'(北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|東京都|神奈川県|新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|大阪府|兵庫県|奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|福岡県|佐賀県|長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県)'
    
    # 次の行がこれらに当てはまる場合は値とみなさない（別のヘッダー）
    header_pattern = r'(氏名|名前|Name|Guest|住所|Address|電話|Tel|Phone|Email|職業|Job|Occupation|Check|Date|No\.|宿泊|人数)'
    
    potential_names = []
    
    for i, line in enumerate(lines):
        # 住所: 都道府県が入っている行は問答無用で住所とする（これが最強）
        if re.search(pref_pattern, line):
            clean_addr = line
            # メールや電話が混ざっていたら消す
            if data["メールアドレス"] in clean_addr: clean_addr = clean_addr.replace(data["メールアドレス"], "")
            if valid_phone and valid_phone in clean_addr: clean_addr = clean_addr.replace(valid_phone, "")
            
            clean_addr = re.sub(r'(住所|Address|住\s*所)[:：\s]*', '', clean_addr, flags=re.IGNORECASE).strip()
            # より長い情報を優先して保存
            if len(clean_addr) > len(data["住所"]):
                data["住所"] = clean_addr
            continue

        # 氏名: ヘッダーを見つけたら「直下の行」を最優先で取得
        if re.search(r'(氏名|名前|Name|Guest)', line, re.IGNORECASE):
            found_name_below = False
            # 直下をチェック
            if i + 1 < len(lines):
                next_line = lines[i+1]
                # 次の行が別のヘッダーっぽくなければ採用
                if not re.search(header_pattern, next_line, re.IGNORECASE) and len(next_line) > 1:
                    potential_names.append(next_line)
                    found_name_below = True
            
            # 直下が取得できなかった（orヘッダーだった）場合のみ、右側を見る
            if not found_name_below:
                val = re.sub(r'(氏名|名前|Name|Guest\s*Name|Guest)[:：\s]*', '', line, flags=re.IGNORECASE).strip()
                if val and len(val) > 1:
                    potential_names.append(val)
        
        # 職業: 同様に「直下の行」を最優先
        if re.search(r'(職業|Occupation|Job)', line, re.IGNORECASE):
            found_job_below = False
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if not re.search(header_pattern, next_line, re.IGNORECASE) and len(next_line) > 1:
                    data["職業"] = next_line
                    found_job_below = True
            
            if not found_job_below:
                val = re.sub(r'(職業|Occupation|Job)[:：\s]*', '', line, flags=re.IGNORECASE).strip()
                if val:
                    data["職業"] = val

        # 年齢: 数字抽出なので、同じ行にあれば採用、なければ次の行から数字を探す
        if re.search(r'(年齢|Age)', line, re.IGNORECASE):
            val = re.sub(r'[^0-9]', '', line)
            if val:
                data["年齢"] = val
            elif i + 1 < len(lines):
                val_next = re.sub(r'[^0-9]', '', lines[i+1])
                if val_next:
                    data["年齢"] = val_next

    # フォールバック: 名前が見つからない場合は、上の方の行を適当に拾う
    if not data["氏名"] and potential_names:
        data["氏名"] = potential_names[0]
    elif not data["氏名"]:
        for line in lines[:3]:
            if re.search(r'(予約|Card|Registration|泊|No\.|Date)', line, re.IGNORECASE): continue
            if data["メールアドレス"] in line: continue
            if len(line) < 2: continue
            data["氏名"] = line
            break

    if data["氏名"]:
        data["氏名"] = data["氏名"].replace("様", "").strip()

    return data

def validate_document_type(text):
    keywords = ["氏名", "名前", "Name", "住所", "Address", "電話", "Tel", "Check-in", "Email"]
    count = 0
    for kw in keywords:
        if kw in text: count += 1
    return count >= 2

def show_custom_success_animation():
    image_path = "assets/nanji_v2.png"
    if not os.path.exists(image_path):
        image_path = "assets/nanji_transparent.png"
    
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            
        st.markdown(f"""
        <style>
        .nanji-floater {{
            position: absolute;
            bottom: -150px;
            background-image: url("data:image/png;base64,{encoded}");
            background-size: contain;
            background-repeat: no-repeat;
            opacity: 0;
            animation-name: floatUp;
            animation-timing-function: ease-in-out; 
            animation-fill-mode: forwards;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        particles = []
        for i in range(25):
            left = random.randint(2, 98)
            size = random.randint(60, 140)
            duration = random.uniform(4.0, 8.0)
            delay = random.uniform(0.0, 3.0)
            p = f'<div class="nanji-floater" style="left:{left}%; width:{size}px; height:{size}px; animation-duration:{duration}s; animation-delay:{delay}s;"></div>'
            particles.append(p)
            
        html_content = f'<div class="floating-container">{"".join(particles)}</div>'
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.balloons()

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
                cols = st.columns(2)
                name = cols[0].text_input("氏名 (A列)", value=data.get("氏名"))
                age = cols[0].text_input("年齢 (B列)", value=data.get("年齢"))
                job = cols[0].text_input("ご職業 (C列)", value=data.get("職業"))
                phone = cols[0].text_input("電話番号 (E列)", value=data.get("電話番号"))
                checkin = cols[1].text_input("チェックイン日 (G列)", value=data.get("チェックイン日"))
                checkout = cols[1].text_input("チェックアウト日 (H列)", value=data.get("チェックアウト日"))
                email = cols[1].text_input("メールアドレス (F列)", value=data.get("メールアドレス"))
                address = st.text_area("住所 (D列)", value=data.get("住所"), height=100)
                
                with st.expander("OCR生データを表示"):
                    st.text_area("解析前のテキスト", st.session_state.get('raw_text', ''), height=150)

                st.markdown("---")
                submitted = st.form_submit_button("✅ 承認してスプレッドシートへ転記")
                if submitted:
                    try:
                        gc = gspread.authorize(creds)
                        sh = gc.open_by_url(SPREADSHEET_URL)
                        ws = sh.get_worksheet(0)
                        row = [name, age, job, address, phone, email, checkin, checkout]
                        ws.append_row(row)
                        try:
                            log_ws = sh.worksheet('OCR_LOG')
                        except:
                            log_ws = sh.add_worksheet(title='OCR_LOG', rows=1000, cols=50)
                            header = ['タイムスタンプ'] + [f'Line {i+1}' for i in range(49)]
                            log_ws.append_row(header)
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        raw_lines = [l.strip() for l in st.session_state.get('raw_text','').splitlines() if l.strip()]
                        log_row = [timestamp] + raw_lines
                        log_ws.update(range_name=f'A{len(log_ws.col_values(1))+1}', values=[log_row])
                        
                        show_custom_success_animation()
                        st.success("✅ 転記完了！（生データログも保存しました）")
                    except Exception as e:
                        st.error(f"書込エラー: {e}")

    st.markdown('<div class="footer">Developed by Center of Okinawa Local Tourism</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
