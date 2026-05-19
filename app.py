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

# FORCE DEPLOY vFinal - Production Stable

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
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.15); }
    h1 { background: -webkit-linear-gradient(45deg, #1a1a1a, #4a4a4a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stFileUploader"] section > div > div > span, [data-testid="stFileUploader"] section > div > div > small { display: none !important; }
    [data-testid="stFileUploader"] section > div > div { display: flex; flex-direction: column; align-items: center; gap: 10px; }
    [data-testid="stFileUploader"] section > div > div::after { content: "ここをタップしてカメラ起動または画像選択"; display: block; order: -1; color: #555; font-weight: bold; margin-top: -10px; margin-bottom: 5px; }
    [data-testid="stFileUploader"] button { color: transparent !important; min-width: 200px; min-height: 50px; position: relative !important; border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; }
    [data-testid="stFileUploader"] button::before { content: "📸 カメラ / 📁 アルバム"; position: absolute; width: 100%; height: 100%; top: 0; left: 0; display: flex; align-items: center; justify-content: center; color: #333 !important; font-size: 1.1rem; font-weight: bold; pointer-events: none; }
    [data-testid="stFileUploader"] button:hover { border-color: #4facfe; }
    @keyframes floatUp {
        0% { bottom: -150px; transform: translateX(0) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        25% { transform: translateX(20px) rotate(5deg); }
        50% { transform: translateX(-20px) rotate(-5deg); }
        75% { transform: translateX(10px) rotate(3deg); opacity: 0.9; }
        100% { bottom: 100vh; transform: translateX(0) rotate(0deg); opacity: 0; }
    }
    .floating-container { position: fixed; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9999; overflow: hidden; }
    .footer { width: 100%; text-align: center; padding: 3rem 0 1rem 0; margin-top: 2rem; color: #888; font-size: 0.85rem; font-family: 'Helvetica Neue', Arial, sans-serif; border-top: 1px solid #e0e0e0; }
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

def perform_ocr_document(image_content, credentials):
    try:
        client = vision.ImageAnnotatorClient(credentials=credentials)
        image = vision.Image(content=image_content)
        # TEXT_DETECTIONに戻す（シンプルな行順序のみ必要）
        image_context = vision.ImageContext(language_hints=["ja", "en"])
        response = client.text_detection(image=image, image_context=image_context)
        if response.error.message:
            st.error(f"OCR Error: {response.error.message}")
            return None
        return response
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def extract_text_content(response):
    if response and response.text_annotations:
        return response.text_annotations[0].description
    return ""

def linear_text_parsing(text):
    """
    OCR生データ（改行区切りテキスト）を上から順番に解析し、
    項目名とその下の値をマッピングする。
    """
    data = {
        "氏名": "", "年齢": "", "職業": "", "住所": "",
        "電話番号": "", "メールアドレス": "", "チェックイン日": "", "チェックアウト日": ""
    }
    
    # 全行リスト（空行除去）
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # 項目判定用Regex
    pat_header_map = {
        "氏名": r'(氏名|名前|Name)',
        "住所": r'(住所|Address|住\s*所)',
        "電話番号": r'(電話|Tel|Phone)',
        "メールアドレス": r'(メール|Email)',
        "職業": r'(職業|Job|Occupation|ご職業)',
        "年齢": r'(年齢|Age)',
        "チェックイン日": r'(チェックイン|Check-in)',
        "チェックアウト日": r'(チェックアウト|Check-out)'
    }
    
    # 都道府県
    pat_pref = r'(北海道|青森県|岩手県|宮城県|秋田県|山形県|福島県|茨城県|栃木県|群馬県|埼玉県|千葉県|東京都|神奈川県|新潟県|富山県|石川県|福井県|山梨県|長野県|岐阜県|静岡県|愛知県|三重県|滋賀県|京都府|大阪府|兵庫県|奈良県|和歌山県|鳥取県|島根県|岡山県|広島県|山口県|徳島県|香川県|愛媛県|高知県|福岡県|佐賀県|長崎県|熊本県|大分県|宮崎県|鹿児島県|沖縄県)'

    # 処理済み行インデックス
    used_indices = set()

    # バリデーション関数定義
    def is_valid_age(text): return re.search(r'\d', text) is not None
    def is_valid_job(text): return not re.match(r'^\d+$', text.strip()) # 数字だけはNG
    def is_valid_phone(text): return len(re.sub(r'\D', '', text)) >= 9
    def is_valid_email(text): return '@' in text
    def is_valid_date(text): return re.search(r'\d{4}', text) is not None

    validators = {
        "年齢": is_valid_age,
        "職業": is_valid_job,
        "電話番号": is_valid_phone,
        "メールアドレス": is_valid_email,
        "チェックイン日": is_valid_date,
        "チェックアウト日": is_valid_date
    }

    # 1. ヘッダー探索ループ
    for i, line in enumerate(lines):
        if i in used_indices: continue
        
        # この行がどのヘッダーにマッチするか
        matched_field = None
        for field, pat in pat_header_map.items():
            if re.search(pat, line, re.IGNORECASE):
                matched_field = field
                break
        
        if matched_field:
            used_indices.add(i)
            
            # 直後から数行先までスキャンして、条件に合う値を探す
            offset = 1
            max_scan = 8 # 探索範囲を少し広げる（メールアドレス対策）
            
            while i + offset < len(lines) and offset < max_scan:
                idx = i + offset
                target_line = lines[idx]
                
                # 自分以外のヘッダーかどうかチェック
                is_other_header = False
                for f, p in pat_header_map.items():
                    if f != matched_field and re.search(p, target_line, re.IGNORECASE):
                        is_other_header = True
                        break
                
                # 他のヘッダーなら、値ではないのでスキップ（探索は続ける：ヘッダーのさらに下に値があるかも）
                if is_other_header:
                    pass 
                elif idx in used_indices:
                    pass # 既に使用済みならスキップ
                else:
                    # ここでバリデーション！
                    is_ok = True
                    if matched_field in validators:
                        if not validators[matched_field](target_line):
                            is_ok = False
                    
                    if is_ok:
                        data[matched_field] = target_line
                        used_indices.add(idx)
                        break # 値が見つかったので探索終了
                
                offset += 1

    # 2. まだ埋まっていない項目をスキャン (住所など)
    if not data["住所"]:
        for i, line in enumerate(lines):
            if i in used_indices: continue
            if re.search(pat_pref, line):
                clean_addr = re.sub(r'(住所|Address|住\s*所)[:：\s]*', '', line).strip()
                data["住所"] = clean_addr
                used_indices.add(i)
                break

    # 3. 電話、メール、日程の補完 (Regex検索)
    full_text = text
    if not data["電話番号"]:
        tels = re.findall(r'0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}', full_text)
        for t in tels:
            if len(re.sub(r'\D','',t)) >= 9:
                data["電話番号"] = t
                break
    
    if not data["メールアドレス"]:
        mails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', full_text)
        if mails: data["メールアドレス"] = mails[0]

    if not data["チェックイン日"]:
        dates = re.findall(r'(\d{4})[\./\-](\d{1,2})[\./\-](\d{1,2})', full_text)
        if len(dates) >= 1:
             data["チェックイン日"] = f"{dates[0][0]}/{dates[0][1]}/{dates[0][2]}"
        if len(dates) >= 2:
             data["チェックアウト日"] = f"{dates[1][0]}/{dates[1][1]}/{dates[1][2]}"

    # クリーニング
    def clean(t): 
        if not t: return ""
        t = re.sub(r'(氏名|名前|住所|電話|メール|職業|年齢|チェックイン|チェックアウト)', '', t).strip()
        t = re.sub(r'^[:：\s]+', '', t).strip()
        return t

    for k in data:
        data[k] = clean(data[k])

    return data

def show_custom_success_animation():
    image_path = "assets/nanji_v2.png"
    if not os.path.exists(image_path): image_path = "assets/nanji_transparent.png"
    if os.path.exists(image_path):
        with open(image_path, "rb") as f: encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""<style>.nanji-floater {{position: absolute; bottom: -150px; background-image: url("data:image/png;base64,{encoded}"); background-size: contain; background-repeat: no-repeat; opacity: 0; animation-name: floatUp; animation-timing-function: ease-in-out; animation-fill-mode: forwards;}}</style>""", unsafe_allow_html=True)
        particles = []
        for i in range(25):
            left, size, dura, delay = random.randint(2, 98), random.randint(60, 140), random.uniform(4.0, 8.0), random.uniform(0.0, 3.0)
            particles.append(f'<div class="nanji-floater" style="left:{left}%; width:{size}px; height:{size}px; animation-duration:{dura}s; animation-delay:{delay}s;"></div>')
        st.markdown(f'<div class="floating-container">{"".join(particles)}</div>', unsafe_allow_html=True)
    else: st.balloons()

def main():
    local_css()
    st.title("📋 予約カードOCR転記システム")
    if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0

    creds = None
    if os.path.exists("service_account.json"):
        creds = load_credentials("service_account.json")
        st.sidebar.success("🔑 認証キー読込済み (Local)")
    elif os.environ.get("GCP_SERVICE_ACCOUNT_JSON"):
        try:
            import json
            creds_dict = json.loads(os.environ.get("GCP_SERVICE_ACCOUNT_JSON"))
            creds = load_credentials(creds_dict)
            st.sidebar.success("🔑 認証キー読込済み (Render Env)")
        except Exception as e:
            st.sidebar.error(f"Render Env Error: {e}")
    elif 'gcp_service_account' in st.secrets:
        try:
            creds = load_credentials(dict(st.secrets['gcp_service_account']))
            st.sidebar.success("🔑 認証キー読込済み (Secrets)")
        except:
            st.sidebar.error("Secrets Error")

    if not creds:
        st.warning("⚠️ 認証キーが見つかりません。環境変数またはSecretsを設定してください。")
        return

    if st.sidebar.button("🔄 リセット / 次の画像を読み込む", type="primary"):
        st.session_state['uploader_key'] += 1
        st.session_state.pop('ocr_result', None)
        st.session_state.pop('raw_text', None)
        st.rerun()

    uploaded_image = st.file_uploader("予約カードを撮影または選択", type=['png', 'jpg', 'jpeg'], key=f"uploader_{st.session_state['uploader_key']}", label_visibility="collapsed")
    
    if uploaded_image:
        final_image = Image.open(uploaded_image)
        col1, col2 = st.columns([1, 1.2]) 
        
        with col1:
            st.subheader("1. 予約カード読込")
            use_enhance = st.checkbox("手書き文字補正を行う (推奨)", value=True, help="文字を濃くし、影を除去して読み取りやすくします。")
            st.image(final_image, caption='読込画像', use_container_width=True)
            
            if st.button("🔍 OCR解析実行", type="primary"):
                with st.spinner('テキスト解析実行中...'):
                    img_byte_arr = io.BytesIO()
                    final_image.save(img_byte_arr, format=final_image.format or 'JPEG')
                    target_bytes = img_byte_arr.getvalue()
                    
                    if use_enhance:
                        target_bytes, processed_cv2_img = preprocess_image(target_bytes)
                        with st.expander("補正後の画像を確認"):
                            st.image(processed_cv2_img, caption="AIが見ている画像", clamp=True, channels='GRAY', use_container_width=True)
                    
                    # 標準TextDetectionに戻す
                    response = perform_ocr_document(target_bytes, creds)
                    
                    if response:
                        # 線形テキスト解析を実行
                        full_text = extract_text_content(response)
                        parsed_data = linear_text_parsing(full_text)
                        
                        st.session_state['ocr_result'] = parsed_data
                        st.session_state['raw_text'] = full_text
                        st.success("解析完了")
                    else:
                        st.error("読み取り失敗")

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
                    if st.form_submit_button("✅ 承認してスプレッドシートへ転記"):
                        st.info("🔄 書き込み処理を開始します...")
                        try:
                            gc = gspread.authorize(creds)
                            sh = gc.open_by_url(SPREADSHEET_URL)
                            
                            # シート名を指定して取得（インデックス0だとずれる可能性があるため）
                            target_sheet_name = 'シート1' 
                            try:
                                ws = sh.worksheet(target_sheet_name)
                            except gspread.WorksheetNotFound:
                                ws = sh.get_worksheet(0)
                                st.warning(f"⚠️ '{target_sheet_name}' が見つかりませんでした。代わりに一番左のシート '{ws.title}' に書き込みます。")
                            
                            st.write(f"書き込み先シート名: {ws.title}")
                            
                            write_data = [name, age, job, address, phone, email, checkin, checkout]
                            st.write(f"書き込みデータを確認: {write_data}")
                            
                            # append_rowだと列がずれる場合があるため、明示的に書き込む
                            # 「上から順に見て、空いている最初の行」を探すロジックに変更
                            col_a = ws.col_values(1)
                            
                            target_row_index = len(col_a) + 1 # デフォルトは末尾
                            
                            # ヘッダー(1行目)があるので、2行目(index 1)からチェック
                            # 途中に空きがあればそこ埋める
                            for i in range(1, len(col_a)):
                                if not col_a[i].strip():
                                    target_row_index = i + 1
                                    break
                            
                            # もしcol_aの長さよりデータの行数がスプレッドシート上で多い場合（途中に空白セルがある場合）
                            # col_valuesは「値のある最後の行」までしか返さないことがあるため、
                            # 念のため get_all_values() でチェックして、本当に空いているか確認するのが確実だが、
                            # 簡易的に「col_valuesで見つけた空き」または「末尾」に書く。
                            # 今回のケース（2-12が空白）なら、col_valuesは1行目までしか返ってこないか、
                            # あるいは13行目まで返ってきて2行目が空文字になっているはず。
                            
                            # col_values が ['氏名', '', '', ..., '山田'] のようになっている場合 -> index 1が見つかる
                            # col_values が ['氏名'] だけの場合 -> 2行目に書く
                            
                            next_row = target_row_index
                            
                            # A列のnext_row行目から書き込み
                            ws.update(range_name=f'A{next_row}', values=[write_data])
                            
                            st.success(f"✅ シート '{ws.title}' の {next_row} 行目に追記しました")
                            
                            try:
                                log_ws = sh.worksheet('OCR_LOG')
                            except:
                                log_ws = sh.add_worksheet(title='OCR_LOG', rows=1000, cols=50)
                                log_ws.append_row(['タイムスタンプ'] + [f'Line {i+1}' for i in range(49)])
                            
                            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            raw_lines = [l.strip() for l in st.session_state.get('raw_text','').splitlines() if l.strip()]
                            log_ws.append_row([ts] + raw_lines)
                            
                            show_custom_success_animation()
                            st.success("✅ 転記完了！（生データログも保存しました）")
                        except Exception as e: 
                            st.error(f"❌ 書き込み中に重大なエラーが発生しました: {type(e).__name__}: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

    st.markdown('<div class="footer">Developed by Center of Okinawa Local Tourism</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
