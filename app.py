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

def get_aligned_card_and_crops(image_bytes):
    """
    画像を読み込み、カードの輪郭を検出して正面の 1000x360 画像に補正。
    その後、9つの入力セルエリアを切り出して返却する。
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h_orig, w_orig = img.shape[:2]

    # 1. 輪郭検出
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    card_contour = None
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < (w_orig * h_orig * 0.15):  # 面積が全体の15%未満は除外
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            if area > max_area:
                card_contour = approx
                max_area = area

    target_w, target_h = 1000, 360
    aligned = None

    # 2. 射影変換
    if card_contour is not None:
        pts = card_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # 右上
        rect[3] = pts[np.argmax(diff)] # 左下
        
        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        aligned = cv2.warpPerspective(img, M, (target_w, target_h))
    else:
        aligned = cv2.resize(img, (target_w, target_h))

    # 3. 9つの手書きセル枠の切り出し座標 (ymin, xmin, ymax, xmax)
    # ※見出し文字を避けて切り出します
    crop_definitions = {
        "氏名": (47, 0, 94, 220),
        "フリガナ": (47, 220, 94, 550),
        "年齢": (47, 550, 94, 780),
        "職業": (47, 780, 94, 1000),
        "住所": (126, 0, 173, 1000),
        "電話番号": (205, 0, 252, 330),
        "メールアドレス": (205, 330, 252, 1000),
        "チェックイン日": (288, 0, 335, 330),
        "チェックアウト日": (288, 330, 335, 1000)
    }

    crops = {}
    for name, (ymin, xmin, ymax, xmax) in crop_definitions.items():
        crop_img = aligned[ymin:ymax, xmin:xmax]
        
        # 切り出したセルごとにコントラスト強調をかけて視認性を上げる
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_crop = clahe.apply(gray_crop)
        
        _, encoded = cv2.imencode('.jpg', enhanced_crop)
        crops[name] = encoded.tobytes()

    return aligned, crops

def perform_ocr_batch(crops_dict, credentials):
    """
    Google Vision API の batch_annotate_images を利用し、
    9つの画像を一度のリクエストでOCR解析する。
    """
    try:
        client = vision.ImageAnnotatorClient(credentials=credentials)
        requests = []
        keys = list(crops_dict.keys())
        
        for key in keys:
            image_content = crops_dict[key]
            image = vision.Image(content=image_content)
            request = vision.AnnotateImageRequest(
                image=image,
                features=[vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)],
                image_context=vision.ImageContext(language_hints=["ja", "en"])
            )
            requests.append(request)
            
        response = client.batch_annotate_images(requests=requests)
        
        parsed_data = {
            "氏名": "", "フリガナ": "", "年齢": "", "職業": "", "住所": "",
            "電話番号": "", "メールアドレス": "", "チェックイン日": "", "チェックアウト日": ""
        }
        raw_texts = []
        
        for idx, res in enumerate(response.responses):
            key = keys[idx]
            if res.error.message:
                print(f"Error on {key}: {res.error.message}")
                continue
                
            text = ""
            if res.text_annotations:
                text = res.text_annotations[0].description.strip()
            
            # 日付セル（チェックイン・アウト）にプリプリントされている「年月日」を取り除くクリーンアップ
            if "日" in key:
                text = re.sub(r'[\s年月日の]+', '/', text).strip('/')
            
            # 全般的なクリーニング
            text = re.sub(r'^[:：\s]+', '', text).strip()
            
            parsed_data[key] = text
            raw_texts.append(f"【{key}】: {text}")
            
        return parsed_data, "\n".join(raw_texts)
        
    except Exception as e:
        st.error(f"API Batch Error: {e}")
        return None, ""

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
            st.image(final_image, caption='読込画像', use_container_width=True)
            
            if st.button("🔍 OCR解析実行", type="primary"):
                with st.spinner('テキスト解析実行中...'):
                    img_byte_arr = io.BytesIO()
                    final_image.save(img_byte_arr, format=final_image.format or 'JPEG')
                    target_bytes = img_byte_arr.getvalue()
                    
                    # 1. 傾き補正およびセル切り出し
                    aligned_img, crops_dict = get_aligned_card_and_crops(target_bytes)
                    
                    # 補正後の画像をUIに表示（確認用）
                    with st.expander("補正後の画像を確認", expanded=True):
                        st.image(aligned_img, caption="補正および規格化されたカード画像", channels='BGR', use_container_width=True)
                    
                    # 2. バッチOCRを実行
                    parsed_data, raw_text_summary = perform_ocr_batch(crops_dict, creds)
                    
                    if parsed_data:
                        st.session_state['ocr_result'] = parsed_data
                        st.session_state['raw_text'] = raw_text_summary
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
                    furigana = cols[0].text_input("フリガナ (非転記)", value=data.get("フリガナ"))
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
                            
                            target_sheet_name = 'シート1' 
                            try:
                                ws = sh.worksheet(target_sheet_name)
                            except gspread.WorksheetNotFound:
                                ws = sh.get_worksheet(0)
                                st.warning(f"⚠️ '{target_sheet_name}' が見つかりませんでした。代わりに一番左のシート '{ws.title}' に書き込みます。")
                            
                            st.write(f"書き込み先シート名: {ws.title}")
                            
                            write_data = [name, age, job, address, phone, email, checkin, checkout]
                            st.write(f"書き込みデータを確認: {write_data}")
                            
                            # 空き行を探す
                            col_a = ws.col_values(1)
                            target_row_index = len(col_a) + 1
                            
                            for i in range(1, len(col_a)):
                                if not col_a[i].strip():
                                    target_row_index = i + 1
                                    break
                            
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
