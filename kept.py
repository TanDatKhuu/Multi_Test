# ==============================================
#           PHẦN 1: CÀI ĐẶT VÀ CÁC HÀM LÕI
# ==============================================

# --- 1. IMPORTS ---
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as MplCircle 
import os
from datetime import datetime
import random
import json
import time # Cần cho animation
import base64
import imageio  # Thêm dòng này
import io       # Thêm dòng này

def html_to_latex(html_string):
    """Chuyển đổi một số thẻ HTML đơn giản sang cú pháp LaTeX."""
    s = html_string.replace('<br>', r' \\ ') # Xuống dòng trong LaTeX
    s = s.replace('<sup>', '^{')
    s = s.replace('</sup>', '}')
    s = s.replace('<sub>', '_{')
    s = s.replace('</sub>', '}')
    s = s.replace('*', r'\cdot ') # Thay dấu * bằng dấu nhân
    return s
	
# Giả định file ảnh và các tài nguyên khác nằm cùng thư mục với script
base_path = os.path.dirname(os.path.abspath(__file__))
# Highlight: Đường dẫn đến thư mục chứa ảnh
FIG_FOLDER = os.path.join(base_path, "fig")

# --- 2. CÁC BIẾN CỐ ĐỊNH VÀ TỪ ĐIỂN NGÔN NGỮ (GIỮ NGUYÊN) ---

# Giả định file ảnh và các tài nguyên khác nằm cùng thư mục với script
base_path = os.path.dirname(os.path.abspath(__file__))
def load_language_file(lang_code):
    """Tải dictionary ngôn ngữ từ file JSON tương ứng."""
    path = os.path.join(base_path, "languages", f"{lang_code}.json")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        # Nếu có lỗi, mặc định trả về tiếng Việt để tránh sập app
        print(f"Error loading language file {path}: {e}")
        path_vi = os.path.join(base_path, "languages", "vi.json")
        with open(path_vi, 'r', encoding='utf-8') as f:
            return json.load(f)
LANG_VI = load_language_file('vi')
LANG_EN = load_language_file('en')

def tr(key):
    return st.session_state.translations.get(key, key)
	
def render_navbar():
    # Sử dụng st.columns để tạo layout cho thanh nav
    # Đã xóa col4 và tỉ lệ của nó (số 1)
    col1, col2, col3, col5 = st.columns([3, 1.5, 1.5, 1.5])

    with col1:
        icon_path_nav = os.path.join(FIG_FOLDER, "icon-app-eureka.png")
        img_tag = ""
        if os.path.exists(icon_path_nav):
            with open(icon_path_nav, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            img_tag = f'<img src="data:image/png;base64,{img_base64}" width="30" style="margin-right: 10px;">'
        st.markdown(f"""
            <div style="display: flex; align-items: center; height: 55px;">
                {img_tag}
                <h3 style='color: #1E3A8A; margin: 0; font-weight: bold;'>MultiStepSim</h3>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        if st.button(tr("nav_home"), use_container_width=True, key="nav_home_btn"):
            st.session_state.page = "welcome"
            st.session_state.welcome_subpage = "home"
            st.rerun()

    # Khối "with col4" đã được xóa hoàn toàn
    
    with col5:
        def on_lang_change():
            selected_display = st.session_state.lang_selector_nav
            lang_options_display = (tr('lang_vi'), tr('lang_en'))
            lang_options_codes = ('vi', 'en')
            selected_index = lang_options_display.index(selected_display)
            st.session_state.lang = lang_options_codes[selected_index]

        lang_options_display = (tr('lang_vi'), tr('lang_en'))
        lang_options_codes = ('vi', 'en')
        current_lang_index = lang_options_codes.index(st.session_state.lang)
        
        st.selectbox(
            "Language", lang_options_display, 
            index=current_lang_index, 
            key='lang_selector_nav', 
            label_visibility="collapsed",
            on_change=on_lang_change
        )
    st.divider()
MODEL_DEFAULTS = {
    "model1": {"O0": 1091.0, "k": 0.073, "t0": 0.0, "t1": 10.0},
    "model2": {"x0": 1.0, "t0": 0.0, "t1": 10.0},
    "model3": {"n": 50.0, "t0": 0.0, "t1": 10.0},
    "model4": {"m": 0.5, "l": 1.0, "a": 0.4, "s": 0.25, "G": 100.0, "Y0": 10.0, "dY0": 5.0, "t0": 0.0, "t1": 10.0},
    "model5": {"x0": 5.0, "y0": 0.0, "u": 2.0, "v": 4.0, "t0": 0.0, "t1": 10.0},
    "model6": {"y_A0": 1.0, "y_B0": 0.0, "y_C0": 0.0, "k1": 1.0, "k2": 2.0, "t0": 0.0, "t1": 1.0},
}
# --- 3. CÁC HÀM TÍNH TOÁN, SOLVERS, MODEL DATA (GIỮ NGUYÊN) ---
# Dán toàn bộ các hàm từ `RK2` đến `_model5_ode_system` và cả dictionary `MODELS_DATA`
# cũng như các class/hàm cho Model 3 (ABM) vào đây.
# Mình sẽ chỉ dán một phần nhỏ làm ví dụ, bạn hãy dán toàn bộ nhé.

# ==============================================
#           ODE Solver Methods
# ==============================================
#RK,AB,AM bậc 1
# Runge-Kutta 2nd order method 
def RK2(f, t, y0):
    h = t[1] - t[0]
    y = [y0]
    for i in range(len(t) - 1):
        k1 = h * f(t[i], y[i])
        k2_std = h * f(t[i] + h, y[i] + k1)
        y_new = y[i] + (k1 + k2_std) / 2.0
        y.append(y_new)
    return np.array(y) 

# Runge-Kutta 3rd order method
def RK3(f, t, y0):
    h = t[1] - t[0]
    y = [y0]
    for i in range(len(t) - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2.0, y[i] + k1 / 2.0)
        k3 = h * f(t[i] + h, y[i] - k1 + 2.0 * k2)
        y_new = y[i] + (k1 + 4.0 * k2 + k3) / 6.0
        y.append(y_new)
    return np.array(y) 

# Runge-Kutta 4th order method
def RK4(f, t, y0):
    h = t[1] - t[0]
    y = [y0]
    for i in range(len(t) - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 2.0, y[i] + k1 / 2.0)
        k3 = h * f(t[i] + h / 2.0, y[i] + k2 / 2.0)
        k4 = h * f(t[i] + h, y[i] + k3)
        y_new = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        y.append(y_new)
    return np.array(y) 

# Runge-Kutta 5th order method
def RK5(f, t, y0):
    h = t[1] - t[0]
    y = [y0]
    for i in range(len(t) - 1):
        k1 = h * f(t[i], y[i])
        k2 = h * f(t[i] + h / 4.0, y[i] + k1 / 4.0)
        k3 = h * f(t[i] + h / 4.0, y[i] + k1 / 8.0 + k2 / 8.0)
        k4 = h * f(t[i] + h / 2.0, y[i] - k2 / 2.0 + k3)
        k5 = h * f(t[i] + 3.0 * h / 4.0, y[i] + 3.0 * k1 / 16.0 + 9.0 * k4 / 16.0)
        k6 = h * f(t[i] + h, y[i] - 3.0 * k1 / 7.0 + 2.0 * k2 / 7.0 + 12.0 * k3 / 7.0 - 12.0 * k4 / 7.0 + 8.0 * k5 / 7.0)
        y_new = y[i] + (7.0 * k1 + 32.0 * k3 + 12.0 * k4 + 32.0 * k5 + 7.0 * k6) / 90.0
        y.append(y_new)
    return np.array(y)

# --- Các hàm ABx, AMx giữ nguyên cách gọi RKx đã sửa ở trên ---
def AB2(f, t, y0):
    h = t[1]-t[0]
    y_start = RK2(f,t[:2],y0) if len(t)>=2 else np.array([y0])
    y = list(y_start)
    for i in range(1, len(t) - 1):
         if i < len(y) and i-1 < len(y):
             y_new = y[i] + h / 2.0 * (3.0 * f(t[i], y[i]) - f(t[i-1], y[i-1]))
             y.append(y_new)
         else:
             print(f"Warning (AB2): Index out of bounds at i={i}, len(y)={len(y)}")
             break 
    return np.array(y)

def AB3(f, t, y0):
    h = t[1]-t[0]
    y_start = RK3(f, t[:3], y0) if len(t) >= 3 else AB2(f, t, y0) 
    y = list(y_start)
    for i in range(2, len(t) - 1):
        if i < len(y) and i-1 < len(y) and i-2 < len(y):
             y_new = y[i] + h / 12.0 * (23.0 * f(t[i], y[i]) - 16.0 * f(t[i-1], y[i-1]) + 5.0 * f(t[i-2], y[i-2]))
             y.append(y_new)
        else:
             print(f"Warning (AB3): Index out of bounds at i={i}, len(y)={len(y)}")
             break
    return np.array(y)

def AB4(f, t, y0):
    h = t[1]-t[0]
    y_start = RK4(f, t[:4], y0) if len(t) >= 4 else AB3(f, t, y0)
    y = list(y_start)
    for i in range(3, len(t) - 1):
         if i < len(y) and i-1 < len(y) and i-2 < len(y) and i-3 < len(y):
             y_new = y[i] + h / 24.0 * (55.0 * f(t[i], y[i]) - 59.0 * f(t[i-1], y[i-1]) + 37.0 * f(t[i-2], y[i-2]) - 9.0 * f(t[i-3], y[i-3]))
             y.append(y_new)
         else:
             print(f"Warning (AB4): Index out of bounds at i={i}, len(y)={len(y)}")
             break
    return np.array(y)

def AB5(f, t, y0):
    h = t[1]-t[0]
    y_start = RK5(f, t[:5], y0) if len(t) >= 5 else AB4(f, t, y0)
    y = list(y_start)
    for i in range(4, len(t) - 1):
         if i < len(y) and i-1 < len(y) and i-2 < len(y) and i-3 < len(y) and i-4 < len(y):
             y_new = y[i] + h / 720.0 * (1901.0 * f(t[i], y[i]) - 2774.0 * f(t[i-1], y[i-1]) + 2616.0 * f(t[i-2], y[i-2]) - 1274.0 * f(t[i-3], y[i-3]) + 251.0 * f(t[i-4], y[i-4]))
             y.append(y_new)
         else:
             print(f"Warning (AB5): Index out of bounds at i={i}, len(y)={len(y)}")
             break
    return np.array(y)

def AM2(f, t, y0):
    h = t[1]-t[0]
    y_start = RK2(f, t[:2], y0) if len(t) >= 2 else np.array([y0])
    y = list(y_start)
    for i in range(1, len(t) - 1):
         if i < len(y) and i-1 < len(y):
             y_pred = y[i] + h / 2.0 * (3.0 * f(t[i], y[i]) - f(t[i-1], y[i-1]))
             if i + 1 < len(t):
                 y_new = y[i] + h / 12.0 * (5.0 * f(t[i+1], y_pred) + 8.0 * f(t[i], y[i]) - f(t[i-1], y[i-1]))
                 y.append(y_new)
             else:
                 print(f"Warning (AM2): Index t[i+1] out of bounds at i={i}")
                 break
         else:
             print(f"Warning (AM2): Index out of bounds at i={i}, len(y)={len(y)}")
             break
    return np.array(y)

def AM3(f, t, y0):
    h = t[1]-t[0]
    y_start = RK3(f, t[:3], y0) if len(t) >= 3 else AM2(f, t, y0)
    y = list(y_start)
    for i in range(2, len(t) - 1):
         if i < len(y) and i-1 < len(y) and i-2 < len(y):
             y_pred = y[i] + h / 12.0 * (23.0 * f(t[i], y[i]) - 16.0 * f(t[i-1], y[i-1]) + 5.0 * f(t[i-2], y[i-2]))
             if i + 1 < len(t):
                 y_new = y[i] + h / 24.0 * (9.0 * f(t[i+1], y_pred) + 19.0 * f(t[i], y[i]) - 5.0 * f(t[i-1], y[i-1]) + f(t[i-2], y[i-2]))
                 y.append(y_new)
             else:
                 print(f"Warning (AM3): Index t[i+1] out of bounds at i={i}")
                 break
         else:
             print(f"Warning (AM3): Index out of bounds at i={i}, len(y)={len(y)}")
             break
    return np.array(y)

def AM4(f, t, y0):
    h = t[1]-t[0]
    y_start = RK4(f, t[:4], y0) if len(t) >= 4 else AM3(f, t, y0)
    y = list(y_start)
    for i in range(3, len(t) - 1):
         if i < len(y) and i-1 < len(y) and i-2 < len(y) and i-3 < len(y):
             y_pred = y[i] + h / 24.0 * (55.0 * f(t[i], y[i]) - 59.0 * f(t[i-1], y[i-1]) + 37.0 * f(t[i-2], y[i-2]) - 9.0 * f(t[i-3], y[i-3]))
             if i + 1 < len(t):
                 y_new = y[i] + h / 720.0 * (251.0 * f(t[i+1], y_pred) + 646.0 * f(t[i], y[i]) - 264.0 * f(t[i-1], y[i-1]) + 106.0 * f(t[i-2], y[i-2]) - 19.0 * f(t[i-3], y[i-3]))
                 y.append(y_new)
             else:
                 print(f"Warning (AM4): Index t[i+1] out of bounds at i={i}")
                 break
         else:
             print(f"Warning (AM4): Index out of bounds at i={i}, len(y)={len(y)}")
             break
    return np.array(y)

#RK,AB,AM bậc 2
# Runge-Kutta 2nd order method for SYSTEMS
def RK2_system(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        k1_vec = F(t, u1[i-1], u2[i-1]) 
        k2_vec = F(t + h, u1[i-1] + h*k1_vec[0], u2[i-1] + h*k1_vec[1])
        u1[i] = u1[i-1] + h/2.0 * (k1_vec[0] + k2_vec[0])
        u2[i] = u2[i-1] + h/2.0 * (k1_vec[1] + k2_vec[1])
    return u1, u2

# Runge-Kutta 3rd order method for SYSTEMS
def RK3_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        k1_vec = F(t, u1[i-1], u2[i-1])
        k2_vec = F(t + h/2.0, u1[i-1] + h/2.0*k1_vec[0], u2[i-1] + h/2.0*k1_vec[1])
        k3_vec = F(t + h, u1[i-1] - h*k1_vec[0] + 2.0*h*k2_vec[0], u2[i-1] - h*k1_vec[1] + 2.0*h*k2_vec[1])
        u1[i] = u1[i-1] + h/6.0 * (k1_vec[0] + 4.0*k2_vec[0] + k3_vec[0])
        u2[i] = u2[i-1] + h/6.0 * (k1_vec[1] + 4.0*k2_vec[1] + k3_vec[1])
    return u1, u2

# Runge-Kutta 4th order method for SYSTEMS
def RK4_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        k1_vec = F(t, u1[i-1], u2[i-1])
        k2_vec = F(t + h/2.0, u1[i-1] + h/2.0*k1_vec[0], u2[i-1] + h/2.0*k1_vec[1])
        k3_vec = F(t + h/2.0, u1[i-1] + h/2.0*k2_vec[0], u2[i-1] + h/2.0*k2_vec[1])
        k4_vec = F(t + h, u1[i-1] + h*k3_vec[0], u2[i-1] + h*k3_vec[1])
        u1[i] = u1[i-1] + h/6.0 * (k1_vec[0] + 2.0*k2_vec[0] + 2.0*k3_vec[0] + k4_vec[0])
        u2[i] = u2[i-1] + h/6.0 * (k1_vec[1] + 2.0*k2_vec[1] + 2.0*k3_vec[1] + k4_vec[1])
    return u1, u2

# Runge-Kutta 5th order method for SYSTEMS (Cash-Karp based coefficients)
def RK5_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(len(t_array) - 1):
        t = t_array[i]
        u1_i = u1[i]
        u2_i = u2[i]
        k1_vec = F(t, u1_i, u2_i)
        k1_u1, k1_u2 = k1_vec[0], k1_vec[1]
        k2_vec = F(t + h / 4.0, u1_i + h / 4.0 * k1_u1, u2_i + h / 4.0 * k1_u2)
        k2_u1, k2_u2 = k2_vec[0], k2_vec[1]
        k3_vec = F(t + h / 4.0, u1_i + h / 8.0 * k1_u1 + h / 8.0 * k2_u1, u2_i + h / 8.0 * k1_u2 + h / 8.0 * k2_u2)
        k3_u1, k3_u2 = k3_vec[0], k3_vec[1]
        k4_vec = F(t + h / 2.0, u1_i - h / 2.0 * k2_u1 + h * k3_u1, u2_i - h / 2.0 * k2_u2 + h * k3_u2)
        k4_u1, k4_u2 = k4_vec[0], k4_vec[1]
        k5_vec = F(t + 3.0 * h / 4.0, u1_i + 3.0 * h / 16.0 * k1_u1 + 9.0 * h / 16.0 * k4_u1, u2_i + 3.0 * h / 16.0 * k1_u2 + 9.0 * h / 16.0 * k4_u2)
        k5_u1, k5_u2 = k5_vec[0], k5_vec[1]
        k6_vec = F(t + h, u1_i - 3.0 * h / 7.0 * k1_u1 + 2.0 * h / 7.0 * k2_u1 + 12.0 * h / 7.0 * k3_u1 - 12.0 * h / 7.0 * k4_u1 + 8.0 * h / 7.0 * k5_u1,
                        u2_i - 3.0 * h / 7.0 * k1_u2 + 2.0 * h / 7.0 * k2_u2 + 12.0 * h / 7.0 * k3_u2 - 12.0 * h / 7.0 * k4_u2 + 8.0 * h / 7.0 * k5_u2)
        k6_u1, k6_u2 = k6_vec[0], k6_vec[1]
        u1[i + 1] = u1_i + h / 90.0 * (7.0 * k1_u1 + 32.0 * k3_u1 + 12.0 * k4_u1 + 32.0 * k5_u1 + 7.0 * k6_u1)
        u2[i + 1] = u2_i + h / 90.0 * (7.0 * k1_u2 + 32.0 * k3_u2 + 12.0 * k4_u2 + 32.0 * k5_u2 + 7.0 * k6_u2)
    return u1, u2

# Adams-Bashforth 2nd order method for SYSTEMS
def AB2_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1_start, u2_start = RK2_system(F, t_array[:2], u10, u20)
    u1[:2] = u1_start
    u2[:2] = u2_start
    for i in range(2, len(t_array)):
        t_prev1 = t_array[i-1]
        t_prev2 = t_array[i-2]
        F_prev1 = F(t_prev1, u1[i-1], u2[i-1])
        F_prev2 = F(t_prev2, u1[i-2], u2[i-2])
        u1[i] = u1[i-1] + h/2.0 * (3.0*F_prev1[0] - F_prev2[0])
        u2[i] = u2[i-1] + h/2.0 * (3.0*F_prev1[1] - F_prev2[1])
    return u1, u2

# Adams-Bashforth 3rd order method for SYSTEMS
def AB3_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1_start, u2_start = RK3_system(F, t_array[:3], u10, u20)
    u1[:3] = u1_start
    u2[:3] = u2_start
    for i in range(3, len(t_array)):
        t_prev1 = t_array[i-1]
        t_prev2 = t_array[i-2]
        t_prev3 = t_array[i-3]
        F_prev1 = F(t_prev1, u1[i-1], u2[i-1])
        F_prev2 = F(t_prev2, u1[i-2], u2[i-2])
        F_prev3 = F(t_prev3, u1[i-3], u2[i-3])
        u1[i] = u1[i-1] + h/12.0 * (23.0*F_prev1[0] - 16.0*F_prev2[0] + 5.0*F_prev3[0])
        u2[i] = u2[i-1] + h/12.0 * (23.0*F_prev1[1] - 16.0*F_prev2[1] + 5.0*F_prev3[1])
    return u1, u2

# Adams-Bashforth 4th order method for SYSTEMS
def AB4_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1_start, u2_start = RK4_system(F, t_array[:4], u10, u20)
    u1[:4] = u1_start
    u2[:4] = u2_start
    for i in range(4, len(t_array)):
        t_prev1 = t_array[i-1]
        t_prev2 = t_array[i-2]
        t_prev3 = t_array[i-3]
        t_prev4 = t_array[i-4]
        F_prev1 = F(t_prev1, u1[i-1], u2[i-1])
        F_prev2 = F(t_prev2, u1[i-2], u2[i-2])
        F_prev3 = F(t_prev3, u1[i-3], u2[i-3])
        F_prev4 = F(t_prev4, u1[i-4], u2[i-4])
        u1[i] = u1[i-1] + h/24.0 * (55.0*F_prev1[0] - 59.0*F_prev2[0] + 37.0*F_prev3[0] - 9.0*F_prev4[0])
        u2[i] = u2[i-1] + h/24.0 * (55.0*F_prev1[1] - 59.0*F_prev2[1] + 37.0*F_prev3[1] - 9.0*F_prev4[1])
    return u1, u2

# Adams-Bashforth 5th order method for SYSTEMS
def AB5_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1_start, u2_start = RK5_system(F, t_array[:5], u10, u20)
    if len(u1_start) < 5: 
         print(f"Warning (AB5_system): RK5 initializer returned fewer than 5 points ({len(u1_start)}). Cannot proceed.")
         return u1_start, u2_start 
    u1[:5] = u1_start
    u2[:5] = u2_start
    for i in range(5, len(t_array)):
        t_prev1 = t_array[i-1]
        t_prev2 = t_array[i-2]
        t_prev3 = t_array[i-3]
        t_prev4 = t_array[i-4]
        t_prev5 = t_array[i-5]
        F_prev1 = F(t_prev1, u1[i-1], u2[i-1])
        F_prev2 = F(t_prev2, u1[i-2], u2[i-2])
        F_prev3 = F(t_prev3, u1[i-3], u2[i-3])
        F_prev4 = F(t_prev4, u1[i-4], u2[i-4])
        F_prev5 = F(t_prev5, u1[i-5], u2[i-5])
        u1[i] = u1[i-1] + h/720.0 * (1901.0*F_prev1[0] - 2774.0*F_prev2[0] + 2616.0*F_prev3[0] - 1274.0*F_prev4[0] + 251.0*F_prev5[0])
        u2[i] = u2[i-1] + h/720.0 * (1901.0*F_prev1[1] - 2774.0*F_prev2[1] + 2616.0*F_prev3[1] - 1274.0*F_prev4[1] + 251.0*F_prev5[1])
    return u1, u2

# Adams-Moulton 2nd order method for SYSTEMS
def AM2_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1_start, u2_start = RK2_system(F, t_array[:2], u10, u20)
    u1[:2] = u1_start
    u2[:2] = u2_start
    for i in range(2, len(t_array)):
        t_curr = t_array[i]
        t_prev1 = t_array[i-1]
        t_prev2 = t_array[i-2]
        F_prev1 = F(t_prev1, u1[i-1], u2[i-1])
        F_prev2 = F(t_prev2, u1[i-2], u2[i-2])
        u1_pred = u1[i-1] + h/2.0 * (3.0*F_prev1[0] - F_prev2[0])
        u2_pred = u2[i-1] + h/2.0 * (3.0*F_prev1[1] - F_prev2[1])
        F_pred = F(t_curr, u1_pred, u2_pred) 
        u1[i] = u1[i-1] + h/12.0*(5.0*F_pred[0] + 8.0*F_prev1[0] - F_prev2[0])
        u2[i] = u2[i-1] + h/12.0*(5.0*F_pred[1] + 8.0*F_prev1[1] - F_prev2[1])
    return u1, u2

# Adams-Moulton 3rd order method for SYSTEMS
def AM3_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1_start, u2_start = RK3_system(F, t_array[:3], u10, u20)
    u1[:3] = u1_start
    u2[:3] = u2_start
    for i in range(3, len(t_array)):
        t_curr = t_array[i]
        t_prev1 = t_array[i-1]
        t_prev2 = t_array[i-2]
        t_prev3 = t_array[i-3]
        F_prev1 = F(t_prev1, u1[i-1], u2[i-1])
        F_prev2 = F(t_prev2, u1[i-2], u2[i-2])
        F_prev3 = F(t_prev3, u1[i-3], u2[i-3])
        u1_pred = u1[i-1] + h/12.0 * (23.0*F_prev1[0] - 16.0*F_prev2[0] + 5.0*F_prev3[0])
        u2_pred = u2[i-1] + h/12.0 * (23.0*F_prev1[1] - 16.0*F_prev2[1] + 5.0*F_prev3[1])
        F_pred = F(t_curr, u1_pred, u2_pred)
        u1[i] = u1[i-1] + h/24.0*(9.0*F_pred[0] + 19.0*F_prev1[0] - 5.0*F_prev2[0] + F_prev3[0])
        u2[i] = u2[i-1] + h/24.0*(9.0*F_pred[1] + 19.0*F_prev1[1] - 5.0*F_prev2[1] + F_prev3[1])
    return u1, u2

# Adams-Moulton 4th order method for SYSTEMS
def AM4_system(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1_start, u2_start = RK4_system(F, t_array[:4], u10, u20)
    u1[:4] = u1_start
    u2[:4] = u2_start
    for i in range(4, len(t_array)):
        t_curr = t_array[i]
        t_prev1 = t_array[i-1]
        t_prev2 = t_array[i-2]
        t_prev3 = t_array[i-3]
        t_prev4 = t_array[i-4]
        F_prev1 = F(t_prev1, u1[i-1], u2[i-1])
        F_prev2 = F(t_prev2, u1[i-2], u2[i-2])
        F_prev3 = F(t_prev3, u1[i-3], u2[i-3])
        F_prev4 = F(t_prev4, u1[i-4], u2[i-4])
        u1_pred = u1[i-1] + h/24.0 * (55.0*F_prev1[0] - 59.0*F_prev2[0] + 37.0*F_prev3[0] - 9.0*F_prev4[0])
        u2_pred = u2[i-1] + h/24.0 * (55.0*F_prev1[1] - 59.0*F_prev2[1] + 37.0*F_prev3[1] - 9.0*F_prev4[1])
        F_pred = F(t_curr, u1_pred, u2_pred)
        u1[i] = u1[i-1] + h/720.0*(251.0*F_pred[0] + 646.0*F_prev1[0] - 264.0*F_prev2[0] + 106.0*F_prev3[0] - 19.0*F_prev4[0])
        u2[i] = u2[i-1] + h/720.0*(251.0*F_pred[1] + 646.0*F_prev1[1] - 264.0*F_prev2[1] + 106.0*F_prev3[1] - 19.0*F_prev4[1])
    return u1, u2

# ==============================================
#   Solver Methods for Model 5 (with break)
# ==============================================

# --- RK Methods with Break ---
def RK2_original_system_M5(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        if not np.isfinite(u1[i-1]) or not np.isfinite(u2[i-1]):
             print(f"RK2_M5 stopping at i={i}: Non-finite input u1={u1[i-1]}, u2={u2[i-1]}")
             return u1[:i], u2[:i]
        try:
            k1_vec = F(t, u1[i-1], u2[i-1])
            if not np.all(np.isfinite(k1_vec)):
                 print(f"RK2_M5 stopping at i={i}: Non-finite k1_vec={k1_vec}")
                 return u1[:i], u2[:i]
            u1_k2_in = u1[i-1] + h*k1_vec[0]
            u2_k2_in = u2[i-1] + h*k1_vec[1]
            if not np.isfinite(u1_k2_in) or not np.isfinite(u2_k2_in):
                 print(f"RK2_M5 stopping at i={i}: Non-finite input for k2")
                 return u1[:i], u2[:i]
            k2_vec = F(t + h, u1_k2_in, u2_k2_in)
            if not np.all(np.isfinite(k2_vec)):
                 print(f"RK2_M5 stopping at i={i}: Non-finite k2_vec={k2_vec}")
                 return u1[:i], u2[:i]
        except Exception as e_fcall:
            print(f"RK2_M5 stopping at i={i}: Error calling F -> {e_fcall}")
            return u1[:i], u2[:i] 
        u1_new = u1[i-1] + h/2.0 * (k1_vec[0] + k2_vec[0])
        u2_new = u2[i-1] + h/2.0 * (k1_vec[1] + k2_vec[1])
        if not np.isfinite(u1_new) or not np.isfinite(u2_new):
            print(f"RK2_M5 stopping at i={i}: Non-finite result u1_new={u1_new}, u2_new={u2_new}")
            return u1[:i], u2[:i]
        u1[i] = u1_new
        u2[i] = u2_new
        if u1[i] <= 0.5:
            return u1[:i+1], u2[:i+1]
    return u1, u2

def RK3_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        if not np.isfinite(u1[i-1]) or not np.isfinite(u2[i-1]): return u1[:i], u2[:i]
        try:
            k1_vec = F(t, u1[i-1], u2[i-1]);                                         
            if not np.all(np.isfinite(k1_vec)): return u1[:i], u2[:i]
            u1_k2_in, u2_k2_in = u1[i-1] + h/2.0*k1_vec[0], u2[i-1] + h/2.0*k1_vec[1]; 
            if not (np.isfinite(u1_k2_in) and np.isfinite(u2_k2_in)): return u1[:i],u2[:i]
            k2_vec = F(t + h/2.0, u1_k2_in, u2_k2_in);                                
            if not np.all(np.isfinite(k2_vec)): return u1[:i], u2[:i]
            u1_k3_in, u2_k3_in = u1[i-1]-h*k1_vec[0]+2.0*h*k2_vec[0], u2[i-1]-h*k1_vec[1]+2.0*h*k2_vec[1]; 
            if not (np.isfinite(u1_k3_in) and np.isfinite(u2_k3_in)): return u1[:i],u2[:i]
            k3_vec = F(t + h, u1_k3_in, u2_k3_in);                                    
            if not np.all(np.isfinite(k3_vec)): return u1[:i], u2[:i]
        except Exception as e_fcall: print(f"RK3_M5 err @ i={i}: {e_fcall}"); return u1[:i], u2[:i]
        u1_new = u1[i-1] + h/6.0 * (k1_vec[0] + 4.0*k2_vec[0] + k3_vec[0])
        u2_new = u2[i-1] + h/6.0 * (k1_vec[1] + 4.0*k2_vec[1] + k3_vec[1])
        if not np.isfinite(u1_new) or not np.isfinite(u2_new): return u1[:i], u2[:i]
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: return u1[:i+1], u2[:i+1]
    return u1, u2

def RK4_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        if not np.isfinite(u1[i-1]) or not np.isfinite(u2[i-1]): return u1[:i], u2[:i]
        try:
            k1_vec = F(t, u1[i-1], u2[i-1]);                                         
            if not np.all(np.isfinite(k1_vec)): 
                return u1[:i], u2[:i]
            u1_k2_in, u2_k2_in = u1[i-1] + h/2.0*k1_vec[0], u2[i-1] + h/2.0*k1_vec[1]; 
            if not (np.isfinite(u1_k2_in) and np.isfinite(u2_k2_in)): 
                return u1[:i],u2[:i]
            k2_vec = F(t + h/2.0, u1_k2_in, u2_k2_in);                                
            if not np.all(np.isfinite(k2_vec)): 
                return u1[:i], u2[:i]
            u1_k3_in, u2_k3_in = u1[i-1] + h/2.0*k2_vec[0], u2[i-1] + h/2.0*k2_vec[1]; 
            if not (np.isfinite(u1_k3_in) and np.isfinite(u2_k3_in)): 
                return u1[:i],u2[:i]
            k3_vec = F(t + h/2.0, u1_k3_in, u2_k3_in);                                
            if not np.all(np.isfinite(k3_vec)): 
                return u1[:i], u2[:i]
            u1_k4_in, u2_k4_in = u1[i-1] + h*k3_vec[0], u2[i-1] + h*k3_vec[1];        
            if not (np.isfinite(u1_k4_in) and np.isfinite(u2_k4_in)): 
                return u1[:i],u2[:i]
            k4_vec = F(t + h, u1_k4_in, u2_k4_in);                                    
            if not np.all(np.isfinite(k4_vec)): 
                return u1[:i], u2[:i]
        except Exception as e_fcall: print(f"RK4_M5 err @ i={i}: {e_fcall}"); return u1[:i], u2[:i]
        u1_new = u1[i-1] + h/6.0 * (k1_vec[0] + 2.0*k2_vec[0] + 2.0*k3_vec[0] + k4_vec[0])
        u2_new = u2[i-1] + h/6.0 * (k1_vec[1] + 2.0*k2_vec[1] + 2.0*k3_vec[1] + k4_vec[1])
        if not np.isfinite(u1_new) or not np.isfinite(u2_new): return u1[:i], u2[:i]
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: return u1[:i+1], u2[:i+1]
    return u1, u2

def RK5_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(len(t_array) - 1):
        t = t_array[i]
        u1_i, u2_i = u1[i], u2[i]
        if not np.isfinite(u1_i) or not np.isfinite(u2_i): return u1[:i], u2[:i]
        try:
            k1_vec=F(t, u1_i, u2_i); k1_u1,k1_u2=k1_vec[0],k1_vec[1]; 
            if not np.all(np.isfinite(k1_vec)): return u1[:i], u2[:i]
            k2_vec=F(t + h/4, u1_i+h/4*k1_u1, u2_i+h/4*k1_u2); k2_u1,k2_u2=k2_vec[0],k2_vec[1]; 
            if not np.all(np.isfinite(k2_vec)): return u1[:i], u2[:i]
            k3_vec=F(t + h/4, u1_i+h/8*k1_u1+h/8*k2_u1, u2_i+h/8*k1_u2+h/8*k2_u2); k3_u1,k3_u2=k3_vec[0],k3_vec[1]; 
            if not np.all(np.isfinite(k3_vec)): return u1[:i], u2[:i]
            k4_vec=F(t + h/2, u1_i-h/2*k2_u1+h*k3_u1, u2_i-h/2*k2_u2+h*k3_u2); k4_u1,k4_u2=k4_vec[0],k4_vec[1]; 
            if not np.all(np.isfinite(k4_vec)): return u1[:i], u2[:i]
            k5_vec=F(t + 3*h/4, u1_i+3*h/16*k1_u1+9*h/16*k4_u1, u2_i+3*h/16*k1_u2+9*h/16*k4_u2); k5_u1,k5_u2=k5_vec[0],k5_vec[1]; 
            if not np.all(np.isfinite(k5_vec)): return u1[:i], u2[:i]
            k6_vec=F(t+h, u1_i-3*h/7*k1_u1+2*h/7*k2_u1+12*h/7*k3_u1-12*h/7*k4_u1+8*h/7*k5_u1, u2_i-3*h/7*k1_u2+2*h/7*k2_u2+12*h/7*k3_u2-12*h/7*k4_u2+8*h/7*k5_u2); k6_u1,k6_u2=k6_vec[0],k6_vec[1]; 
            if not np.all(np.isfinite(k6_vec)): return u1[:i], u2[:i]
        except Exception as e_fcall: print(f"RK5_M5 err @ i={i}: {e_fcall}"); return u1[:i], u2[:i]
        u1_new = u1_i + h/90.0*(7*k1_u1+32*k3_u1+12*k4_u1+32*k5_u1+7*k6_u1)
        u2_new = u2_i + h/90.0*(7*k1_u2+32*k3_u2+12*k4_u2+32*k5_u2+7*k6_u2)
        if not np.isfinite(u1_new) or not np.isfinite(u2_new): return u1[:i], u2[:i]
        u1[i+1], u2[i+1] = u1_new, u2_new
        if u1[i+1] <= 0.5: return u1[:i+2], u2[:i+2] 
    return u1, u2

# --- AB Methods with Break ---
def AB2_original_system_M5(F, t_array, u10, u20):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0], u2[0] = u10, u20
    try:
        u1_start, u2_start = RK2_system_M5(F, t_array[:2], u10, u20) # Gọi bản M5
        start_len = len(u1_start)
        if start_len < 2: return u1[:start_len], u2[:start_len]
        u1[:start_len] = u1_start
        u2[:start_len] = u2_start
    except Exception as e_init: print(f"AB2_M5 init err: {e_init}"); return u1[:1], u2[:1]
    actual_len = len(t_array)
    for i in range(2, len(t_array)): 
        t_prev1, u1_prev1, u2_prev1 = t_array[i-1], u1[i-1], u2[i-1]
        t_prev2, u1_prev2, u2_prev2 = t_array[i-2], u1[i-2], u2[i-2]
        if not (np.isfinite(u1_prev1) and np.isfinite(u2_prev1) and np.isfinite(u1_prev2) and np.isfinite(u2_prev2)): actual_len=i; break
        try:
            F_prev1 = F(t_prev1, u1_prev1, u2_prev1); 
            if not np.all(np.isfinite(F_prev1)): actual_len=i; break
            F_prev2 = F(t_prev2, u1_prev2, u2_prev2); 
            if not np.all(np.isfinite(F_prev2)): actual_len=i; break
        except Exception as e_fcall: print(f"AB2_M5 F err @ i={i}: {e_fcall}"); actual_len=i; break
        u1_new = u1_prev1 + h/2.0 * (3.0*F_prev1[0] - F_prev2[0])
        u2_new = u2_prev1 + h/2.0 * (3.0*F_prev1[1] - F_prev2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AB3_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK3_system_M5(F, t_array[:3], u10, u20); start_len=len(u1_start)
        if start_len < 3: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AB3_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(3, len(t_array)):
        t0, x0, y0 = t_array[i-1], u1[i-1], u2[i-1]
        t1, x1, y1 = t_array[i-2], u1[i-2], u2[i-2]
        t2, x2, y2 = t_array[i-3], u1[i-3], u2[i-3]
        if not all(np.isfinite(v) for v in [x0, y0, x1, y1, x2, y2]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2)
            if not (np.all(np.isfinite(F0)) and np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))): actual_len=i; break
        except Exception as e: print(f"AB3_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/12.0 * (23.0*F0[0] - 16.0*F1[0] + 5.0*F2[0])
        u2_new = y0 + h/12.0 * (23.0*F0[1] - 16.0*F1[1] + 5.0*F2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AB4_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK4_system_M5(F, t_array[:4], u10, u20); start_len=len(u1_start)
        if start_len < 4: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AB4_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(4, len(t_array)):
        t0,x0,y0=t_array[i-1],u1[i-1],u2[i-1]; t1,x1,y1=t_array[i-2],u1[i-2],u2[i-2]
        t2,x2,y2=t_array[i-3],u1[i-3],u2[i-3]; t3,x3,y3=t_array[i-4],u1[i-4],u2[i-4]
        if not all(np.isfinite(v) for v in [x0,y0,x1,y1,x2,y2,x3,y3]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2); F3=F(t3,x3,y3)
            if not all(np.all(np.isfinite(f)) for f in [F0, F1, F2, F3]): actual_len=i; break
        except Exception as e: print(f"AB4_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/24.0 * (55.0*F0[0] - 59.0*F1[0] + 37.0*F2[0] - 9.0*F3[0])
        u2_new = y0 + h/24.0 * (55.0*F0[1] - 59.0*F1[1] + 37.0*F2[1] - 9.0*F3[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AB5_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK5_system_M5(F, t_array[:5], u10, u20); start_len=len(u1_start)
        if start_len < 5: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AB5_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(5, len(t_array)):
        t0,x0,y0=t_array[i-1],u1[i-1],u2[i-1]; t1,x1,y1=t_array[i-2],u1[i-2],u2[i-2]
        t2,x2,y2=t_array[i-3],u1[i-3],u2[i-3]; t3,x3,y3=t_array[i-4],u1[i-4],u2[i-4]
        t4,x4,y4=t_array[i-5],u1[i-5],u2[i-5]
        if not all(np.isfinite(v) for v in [x0,y0,x1,y1,x2,y2,x3,y3,x4,y4]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2); F3=F(t3,x3,y3); F4=F(t4,x4,y4)
            if not all(np.all(np.isfinite(f)) for f in [F0, F1, F2, F3, F4]): actual_len=i; break
        except Exception as e: print(f"AB5_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/720.0*(1901*F0[0]-2774*F1[0]+2616*F2[0]-1274*F3[0]+251*F4[0])
        u2_new = y0 + h/720.0*(1901*F0[1]-2774*F1[1]+2616*F2[1]-1274*F3[1]+251*F4[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

# --- AM Methods with Break ---
def AM2_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK2_system_M5(F, t_array[:2], u10, u20); start_len=len(u1_start)
        if start_len < 2: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AM2_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(2, len(t_array)): 
        t_curr = t_array[i] 
        t_prev1, u1_prev1, u2_prev1 = t_array[i-1], u1[i-1], u2[i-1]
        t_prev2, u1_prev2, u2_prev2 = t_array[i-2], u1[i-2], u2[i-2]
        if not all(np.isfinite(v) for v in [u1_prev1, u2_prev1, u1_prev2, u2_prev2]): actual_len=i; break
        try:
            F_prev1 = F(t_prev1, u1_prev1, u2_prev1); 
            if not np.all(np.isfinite(F_prev1)): actual_len=i; break
            F_prev2 = F(t_prev2, u1_prev2, u2_prev2); 
            if not np.all(np.isfinite(F_prev2)): actual_len=i; break
        except Exception as e: print(f"AM2_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_pred = u1_prev1 + h/2.0 * (3.0*F_prev1[0] - F_prev2[0])
        u2_pred = u2_prev1 + h/2.0 * (3.0*F_prev1[1] - F_prev2[1])
        if not (np.isfinite(u1_pred) and np.isfinite(u2_pred)): actual_len=i; break
        try:
            F_pred = F(t_curr, u1_pred, u2_pred); 
            if not np.all(np.isfinite(F_pred)): actual_len=i; break
        except Exception as e: print(f"AM2_M5 F_pred err @ i={i}: {e}"); actual_len=i; break
        u1_new = u1_prev1 + h/12.0*(5.0*F_pred[0] + 8.0*F_prev1[0] - F_prev2[0])
        u2_new = u2_prev1 + h/12.0*(5.0*F_pred[1] + 8.0*F_prev1[1] - F_prev2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AM3_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK3_system_M5(F, t_array[:3], u10, u20); start_len=len(u1_start)
        if start_len < 3: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AM3_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(3, len(t_array)):
        t_curr = t_array[i]
        t0, x0, y0 = t_array[i-1], u1[i-1], u2[i-1]
        t1, x1, y1 = t_array[i-2], u1[i-2], u2[i-2]
        t2, x2, y2 = t_array[i-3], u1[i-3], u2[i-3]
        if not all(np.isfinite(v) for v in [x0, y0, x1, y1, x2, y2]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2)
            if not (np.all(np.isfinite(F0)) and np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))): actual_len=i; break
        except Exception as e: print(f"AM3_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_pred = x0 + h/12.0 * (23.0*F0[0] - 16.0*F1[0] + 5.0*F2[0])
        u2_pred = y0 + h/12.0 * (23.0*F0[1] - 16.0*F1[1] + 5.0*F2[1])
        if not (np.isfinite(u1_pred) and np.isfinite(u2_pred)): actual_len=i; break
        try:
            F_pred = F(t_curr, u1_pred, u2_pred); 
            if not np.all(np.isfinite(F_pred)): actual_len=i; break
        except Exception as e: print(f"AM3_M5 F_pred err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/24.0*(9.0*F_pred[0] + 19.0*F0[0] - 5.0*F1[0] + F2[0])
        u2_new = y0 + h/24.0*(9.0*F_pred[1] + 19.0*F0[1] - 5.0*F1[1] + F2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AM4_original_system_M5(F, t_array, u10, u20): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK4_system_M5(F, t_array[:4], u10, u20); start_len=len(u1_start)
        if start_len < 4: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AM4_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(4, len(t_array)):
        t_curr=t_array[i]
        t0,x0,y0=t_array[i-1],u1[i-1],u2[i-1]; t1,x1,y1=t_array[i-2],u1[i-2],u2[i-2]
        t2,x2,y2=t_array[i-3],u1[i-3],u2[i-3]; t3,x3,y3=t_array[i-4],u1[i-4],u2[i-4]
        if not all(np.isfinite(v) for v in [x0,y0,x1,y1,x2,y2,x3,y3]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2); F3=F(t3,x3,y3)
            if not all(np.all(np.isfinite(f)) for f in [F0, F1, F2, F3]): actual_len=i; break
        except Exception as e: print(f"AM4_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_pred = x0 + h/24.0 * (55.0*F0[0] - 59.0*F1[0] + 37.0*F2[0] - 9.0*F3[0])
        u2_pred = y0 + h/24.0 * (55.0*F0[1] - 59.0*F1[1] + 37.0*F2[1] - 9.0*F3[1])
        if not (np.isfinite(u1_pred) and np.isfinite(u2_pred)): actual_len=i; break
        try:
            F_pred = F(t_curr, u1_pred, u2_pred); 
            if not np.all(np.isfinite(F_pred)): actual_len=i; break
        except Exception as e: print(f"AM4_M5 F_pred err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/720.0*(251.0*F_pred[0] + 646.0*F0[0] - 264.0*F1[0] + 106.0*F2[0] - 19.0*F3[0])
        u2_new = y0 + h/720.0*(251.0*F_pred[1] + 646.0*F0[1] - 264.0*F1[1] + 106.0*F2[1] - 19.0*F3[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.5: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

# ==============================================
#   NEW Solver Methods for Model 5 - SIMULATION 1
# ==============================================

# --- RK Methods with Break (Simplified Break for RK initializers) ---
def RK2_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        if not np.isfinite(u1[i-1]) or not np.isfinite(u2[i-1]):
             print(f"RK2_M5 stopping at i={i}: Non-finite input u1={u1[i-1]}, u2={u2[i-1]}")
             return u1[:i], u2[:i]
        try:
            k1_vec = F(t, u1[i-1], u2[i-1])
            if not np.all(np.isfinite(k1_vec)):
                 print(f"RK2_M5 stopping at i={i}: Non-finite k1_vec={k1_vec}")
                 return u1[:i], u2[:i]
            u1_k2_in = u1[i-1] + h*k1_vec[0]
            u2_k2_in = u2[i-1] + h*k1_vec[1]
            if not np.isfinite(u1_k2_in) or not np.isfinite(u2_k2_in):
                 print(f"RK2_M5 stopping at i={i}: Non-finite input for k2")
                 return u1[:i], u2[:i]
            k2_vec = F(t + h, u1_k2_in, u2_k2_in)
            if not np.all(np.isfinite(k2_vec)):
                 print(f"RK2_M5 stopping at i={i}: Non-finite k2_vec={k2_vec}")
                 return u1[:i], u2[:i]
        except Exception as e_fcall:
            print(f"RK2_M5 stopping at i={i}: Error calling F -> {e_fcall}")
            return u1[:i], u2[:i]
        u1_new = u1[i-1] + h/2.0 * (k1_vec[0] + k2_vec[0])
        u2_new = u2[i-1] + h/2.0 * (k1_vec[1] + k2_vec[1])
        if not np.isfinite(u1_new) or not np.isfinite(u2_new):
            print(f"RK2_M5 stopping at i={i}: Non-finite result u1_new={u1_new}, u2_new={u2_new}")
            return u1[:i], u2[:i]
        u1[i] = u1_new
        u2[i] = u2_new  
        if u1[i] <= 0.01: 
            return u1[:i+1], u2[:i+1]
    return u1, u2

def RK3_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        if not np.isfinite(u1[i-1]) or not np.isfinite(u2[i-1]): return u1[:i], u2[:i]
        try:
            k1_vec = F(t, u1[i-1], u2[i-1]);
            if not np.all(np.isfinite(k1_vec)): return u1[:i], u2[:i]
            u1_k2_in, u2_k2_in = u1[i-1] + h/2.0*k1_vec[0], u2[i-1] + h/2.0*k1_vec[1];
            if not (np.isfinite(u1_k2_in) and np.isfinite(u2_k2_in)): return u1[:i],u2[:i]
            k2_vec = F(t + h/2.0, u1_k2_in, u2_k2_in);
            if not np.all(np.isfinite(k2_vec)): return u1[:i], u2[:i]
            u1_k3_in, u2_k3_in = u1[i-1]-h*k1_vec[0]+2.0*h*k2_vec[0], u2[i-1]-h*k1_vec[1]+2.0*h*k2_vec[1];
            if not (np.isfinite(u1_k3_in) and np.isfinite(u2_k3_in)): return u1[:i],u2[:i]
            k3_vec = F(t + h, u1_k3_in, u2_k3_in);
            if not np.all(np.isfinite(k3_vec)): return u1[:i], u2[:i]
        except Exception as e_fcall: print(f"RK3_M5 err @ i={i}: {e_fcall}"); return u1[:i], u2[:i]

        u1_new = u1[i-1] + h/6.0 * (k1_vec[0] + 4.0*k2_vec[0] + k3_vec[0])
        u2_new = u2[i-1] + h/6.0 * (k1_vec[1] + 4.0*k2_vec[1] + k3_vec[1])
        if not np.isfinite(u1_new) or not np.isfinite(u2_new): return u1[:i], u2[:i]
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.01: return u1[:i+1], u2[:i+1]
    return u1, u2

def RK4_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None): 
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(1, len(t_array)):
        t = t_array[i-1]
        if not np.isfinite(u1[i-1]) or not np.isfinite(u2[i-1]): return u1[:i], u2[:i]
        try:
            k1_vec = F(t, u1[i-1], u2[i-1]);
            if not np.all(np.isfinite(k1_vec)): return u1[:i], u2[:i]
            u1_k2_in, u2_k2_in = u1[i-1] + h/2.0*k1_vec[0], u2[i-1] + h/2.0*k1_vec[1];
            if not (np.isfinite(u1_k2_in) and np.isfinite(u2_k2_in)): return u1[:i],u2[:i]
            k2_vec = F(t + h/2.0, u1_k2_in, u2_k2_in);
            if not np.all(np.isfinite(k2_vec)): return u1[:i], u2[:i]
            u1_k3_in, u2_k3_in = u1[i-1] + h/2.0*k2_vec[0], u2[i-1] + h/2.0*k2_vec[1];
            if not (np.isfinite(u1_k3_in) and np.isfinite(u2_k3_in)): return u1[:i],u2[:i]
            k3_vec = F(t + h/2.0, u1_k3_in, u2_k3_in);
            if not np.all(np.isfinite(k3_vec)): return u1[:i], u2[:i]
            u1_k4_in, u2_k4_in = u1[i-1] + h*k3_vec[0], u2[i-1] + h*k3_vec[1];
            if not (np.isfinite(u1_k4_in) and np.isfinite(u2_k4_in)): return u1[:i],u2[:i]
            k4_vec = F(t + h, u1_k4_in, u2_k4_in);
            if not np.all(np.isfinite(k4_vec)): return u1[:i], u2[:i]
        except Exception as e_fcall: print(f"RK4_M5 err @ i={i}: {e_fcall}"); return u1[:i], u2[:i]
        u1_new = u1[i-1] + h/6.0 * (k1_vec[0] + 2.0*k2_vec[0] + 2.0*k3_vec[0] + k4_vec[0])
        u2_new = u2[i-1] + h/6.0 * (k1_vec[1] + 2.0*k2_vec[1] + 2.0*k3_vec[1] + k4_vec[1])
        if not np.isfinite(u1_new) or not np.isfinite(u2_new): return u1[:i], u2[:i]
        u1[i], u2[i] = u1_new, u2_new
        if u1[i] <= 0.01: return u1[:i+1], u2[:i+1]
    return u1, u2

def RK5_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0] = u10
    u2[0] = u20
    for i in range(len(t_array) - 1): 
        t = t_array[i]
        u1_i, u2_i = u1[i], u2[i]
        if not np.isfinite(u1_i) or not np.isfinite(u2_i):
            print(f"RK5_M5 stopping at step i={i} due to non-finite input u1_i, u2_i.")
            return u1[:i], u2[:i] 
        try:
            k1_vec = F(t, u1_i, u2_i)
            if not np.all(np.isfinite(k1_vec)): return u1[:i], u2[:i]
            k1_u1, k1_u2 = k1_vec[0], k1_vec[1]
            k2_input_u1 = u1_i + h / 4.0 * k1_u1
            k2_input_u2 = u2_i + h / 4.0 * k1_u2
            if not (np.isfinite(k2_input_u1) and np.isfinite(k2_input_u2)): return u1[:i], u2[:i]
            k2_vec = F(t + h / 4.0, k2_input_u1, k2_input_u2)
            if not np.all(np.isfinite(k2_vec)): return u1[:i], u2[:i]
            k2_u1, k2_u2 = k2_vec[0], k2_vec[1]
            k3_input_u1 = u1_i + h / 8.0 * k1_u1 + h / 8.0 * k2_u1
            k3_input_u2 = u2_i + h / 8.0 * k1_u2 + h / 8.0 * k2_u2
            if not (np.isfinite(k3_input_u1) and np.isfinite(k3_input_u2)): return u1[:i], u2[:i]
            k3_vec = F(t + h / 4.0, k3_input_u1, k3_input_u2)
            if not np.all(np.isfinite(k3_vec)): return u1[:i], u2[:i]
            k3_u1, k3_u2 = k3_vec[0], k3_vec[1]
            k4_input_u1 = u1_i - h / 2.0 * k2_u1 + h * k3_u1
            k4_input_u2 = u2_i - h / 2.0 * k2_u2 + h * k3_u2
            if not (np.isfinite(k4_input_u1) and np.isfinite(k4_input_u2)): return u1[:i], u2[:i]
            k4_vec = F(t + h / 2.0, k4_input_u1, k4_input_u2)
            if not np.all(np.isfinite(k4_vec)): return u1[:i], u2[:i]
            k4_u1, k4_u2 = k4_vec[0], k4_vec[1]
            k5_input_u1 = u1_i + 3.0 * h / 16.0 * k1_u1 + 9.0 * h / 16.0 * k4_u1
            k5_input_u2 = u2_i + 3.0 * h / 16.0 * k1_u2 + 9.0 * h / 16.0 * k4_u2
            if not (np.isfinite(k5_input_u1) and np.isfinite(k5_input_u2)): return u1[:i], u2[:i]
            k5_vec = F(t + 3.0 * h / 4.0, k5_input_u1, k5_input_u2)
            if not np.all(np.isfinite(k5_vec)): return u1[:i], u2[:i]
            k5_u1, k5_u2 = k5_vec[0], k5_vec[1]
            k6_input_u1 = u1_i - 3.0*h/7.0*k1_u1 + 2.0*h/7.0*k2_u1 + 12.0*h/7.0*k3_u1 - 12.0*h/7.0*k4_u1 + 8.0*h/7.0*k5_u1
            k6_input_u2 = u2_i - 3.0*h/7.0*k1_u2 + 2.0*h/7.0*k2_u2 + 12.0*h/7.0*k3_u2 - 12.0*h/7.0*k4_u2 + 8.0*h/7.0*k5_u2
            if not (np.isfinite(k6_input_u1) and np.isfinite(k6_input_u2)): return u1[:i], u2[:i]
            k6_vec = F(t + h, k6_input_u1, k6_input_u2)
            if not np.all(np.isfinite(k6_vec)): return u1[:i], u2[:i]
            k6_u1, k6_u2 = k6_vec[0], k6_vec[1]
        except Exception as e_fcall:
            print(f"RK5_M5 stopping at step i={i}: Error calling F -> {e_fcall}")
            return u1[:i], u2[:i]
        u1_new_rk5 = u1_i + h / 90.0 * (7.0 * k1_u1 + 32.0 * k3_u1 + 12.0 * k4_u1 + 32.0 * k5_u1 + 7.0 * k6_u1)
        u2_new_rk5 = u2_i + h / 90.0 * (7.0 * k1_u2 + 32.0 * k3_u2 + 12.0 * k4_u2 + 32.0 * k5_u2 + 7.0 * k6_u2)
        if not np.isfinite(u1_new_rk5) or not np.isfinite(u2_new_rk5):
            print(f"RK5_M5 stopping at step i={i}: Non-finite result u1_new, u2_new.")
            return u1[:i], u2[:i] 
        u1[i+1] = u1_new_rk5
        u2[i+1] = u2_new_rk5
        new_z_vec_for_break_rk5 = np.array([u1[i+1], u2[i+1]])
        break_occurred_rk5 = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_rk5[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_rk5) < 0.5:
                    break_occurred_rk5 = True
            else:
                if np.linalg.norm(new_z_vec_for_break_rk5) < 0.001:
                    return u1[:i+1], u2[:i+1] 
            if break_occurred_rk5:
                return u1[:i+2], u2[:i+2] 
        else:
            if u1[i+1] <= 0.01: 
                return u1[:i+2], u2[:i+2] 
    return u1, u2

def AB2_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array))
    u2 = np.zeros(len(t_array))
    u1[0], u2[0] = u10, u20
    try:
        u1_start, u2_start = RK2_system_M5(F, t_array[:2], u10, u20, v_t_param, v_n_param, d_param)
        start_len = len(u1_start)
        if start_len < 2: return u1[:start_len], u2[:start_len]
        u1[:start_len] = u1_start
        u2[:start_len] = u2_start
    except Exception as e_init: print(f"AB2_M5 init err: {e_init}"); return u1[:1], u2[:1]
    actual_len = len(t_array)
    for i in range(2, len(t_array)):
        t_prev1, u1_prev1, u2_prev1 = t_array[i-1], u1[i-1], u2[i-1]
        t_prev2, u1_prev2, u2_prev2 = t_array[i-2], u1[i-2], u2[i-2]
        if not (np.isfinite(u1_prev1) and np.isfinite(u2_prev1) and np.isfinite(u1_prev2) and np.isfinite(u2_prev2)):
            actual_len=i; break
        try:
            F_prev1 = F(t_prev1, u1_prev1, u2_prev1);
            if not np.all(np.isfinite(F_prev1)): actual_len=i; break
            F_prev2 = F(t_prev2, u1_prev2, u2_prev2);
            if not np.all(np.isfinite(F_prev2)): actual_len=i; break
        except Exception as e_fcall: print(f"AB2_M5 F err @ i={i}: {e_fcall}"); actual_len=i; break
        u1_new = u1_prev1 + h/2.0 * (3.0*F_prev1[0] - F_prev2[0])
        u2_new = u2_prev1 + h/2.0 * (3.0*F_prev1[1] - F_prev2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        new_z_vec_for_break_ab = np.array([u1[i], u2[i]])
        break_occurred_ab = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_ab[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_ab) < 0.5:
                    actual_len = i + 1 
                    break_occurred_ab = True
            else:
                if np.linalg.norm(new_z_vec_for_break_ab) < 0.001:
                    actual_len = i 
                    break_occurred_ab = True
            if break_occurred_ab: break
        else: 
            if u1[i] <= 0.01: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AB3_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK3_system_M5(F, t_array[:3], u10, u20, v_t_param, v_n_param, d_param)
        start_len=len(u1_start)
        if start_len < 3: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AB3_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(3, len(t_array)):
        t0, x0, y0 = t_array[i-1], u1[i-1], u2[i-1]
        t1, x1, y1 = t_array[i-2], u1[i-2], u2[i-2]
        t2, x2, y2 = t_array[i-3], u1[i-3], u2[i-3]
        if not all(np.isfinite(v) for v in [x0, y0, x1, y1, x2, y2]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2)
            if not (np.all(np.isfinite(F0)) and np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))): actual_len=i; break
        except Exception as e: print(f"AB3_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/12.0 * (23.0*F0[0] - 16.0*F1[0] + 5.0*F2[0])
        u2_new = y0 + h/12.0 * (23.0*F0[1] - 16.0*F1[1] + 5.0*F2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        new_z_vec_for_break_ab = np.array([u1[i], u2[i]])
        break_occurred_ab = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_ab[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_ab) < 0.5:
                    actual_len = i + 1; break_occurred_ab = True
            else:
                if np.linalg.norm(new_z_vec_for_break_ab) < 0.001:
                    actual_len = i; break_occurred_ab = True
            if break_occurred_ab: break
        else:
            if u1[i] <= 0.01: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AB4_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK4_system_M5(F, t_array[:4], u10, u20, v_t_param, v_n_param, d_param)
        start_len=len(u1_start)
        if start_len < 4: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AB4_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(4, len(t_array)):
        t0,x0,y0=t_array[i-1],u1[i-1],u2[i-1]; t1,x1,y1=t_array[i-2],u1[i-2],u2[i-2]
        t2,x2,y2=t_array[i-3],u1[i-3],u2[i-3]; t3,x3,y3=t_array[i-4],u1[i-4],u2[i-4]
        if not all(np.isfinite(v) for v in [x0,y0,x1,y1,x2,y2,x3,y3]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2); F3=F(t3,x3,y3)
            if not all(np.all(np.isfinite(f)) for f in [F0, F1, F2, F3]): actual_len=i; break
        except Exception as e: print(f"AB4_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/24.0 * (55.0*F0[0] - 59.0*F1[0] + 37.0*F2[0] - 9.0*F3[0])
        u2_new = y0 + h/24.0 * (55.0*F0[1] - 59.0*F1[1] + 37.0*F2[1] - 9.0*F3[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        new_z_vec_for_break_ab = np.array([u1[i], u2[i]])
        break_occurred_ab = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_ab[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_ab) < 0.5:
                    actual_len = i + 1; break_occurred_ab = True
            else:
                if np.linalg.norm(new_z_vec_for_break_ab) < 0.001:
                    actual_len = i; break_occurred_ab = True
            if break_occurred_ab: break
        else:
            if u1[i] <= 0.01: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AB5_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK5_system_M5(F, t_array[:5], u10, u20, v_t_param, v_n_param, d_param)
        start_len=len(u1_start)
        if start_len < 5: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AB5_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(5, len(t_array)):
        t0,x0,y0=t_array[i-1],u1[i-1],u2[i-1]; t1,x1,y1=t_array[i-2],u1[i-2],u2[i-2]
        t2,x2,y2=t_array[i-3],u1[i-3],u2[i-3]; t3,x3,y3=t_array[i-4],u1[i-4],u2[i-4]
        t4,x4,y4=t_array[i-5],u1[i-5],u2[i-5]
        if not all(np.isfinite(v) for v in [x0,y0,x1,y1,x2,y2,x3,y3,x4,y4]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2); F3=F(t3,x3,y3); F4=F(t4,x4,y4)
            if not all(np.all(np.isfinite(f)) for f in [F0, F1, F2, F3, F4]): actual_len=i; break
        except Exception as e: print(f"AB5_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/720.0*(1901*F0[0]-2774*F1[0]+2616*F2[0]-1274*F3[0]+251*F4[0])
        u2_new = y0 + h/720.0*(1901*F0[1]-2774*F1[1]+2616*F2[1]-1274*F3[1]+251*F4[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        new_z_vec_for_break_ab = np.array([u1[i], u2[i]])
        break_occurred_ab = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_ab[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_ab) < 0.5:
                    actual_len = i + 1; break_occurred_ab = True
            else:
                if np.linalg.norm(new_z_vec_for_break_ab) < 0.001:
                    actual_len = i; break_occurred_ab = True
            if break_occurred_ab: break
        else:
            if u1[i] <= 0.01: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AM2_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK2_system_M5(F, t_array[:2], u10, u20, v_t_param, v_n_param, d_param)
        start_len=len(u1_start)
        if start_len < 2: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AM2_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(2, len(t_array)):
        t_curr = t_array[i]
        t_prev1, u1_prev1, u2_prev1 = t_array[i-1], u1[i-1], u2[i-1]
        t_prev2, u1_prev2, u2_prev2 = t_array[i-2], u1[i-2], u2[i-2]
        if not all(np.isfinite(v) for v in [u1_prev1, u2_prev1, u1_prev2, u2_prev2]): actual_len=i; break
        try:
            F_prev1 = F(t_prev1, u1_prev1, u2_prev1);
            if not np.all(np.isfinite(F_prev1)): actual_len=i; break
            F_prev2 = F(t_prev2, u1_prev2, u2_prev2);
            if not np.all(np.isfinite(F_prev2)): actual_len=i; break
        except Exception as e: print(f"AM2_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_pred = u1_prev1 + h/2.0 * (3.0*F_prev1[0] - F_prev2[0])
        u2_pred = u2_prev1 + h/2.0 * (3.0*F_prev1[1] - F_prev2[1])
        if not (np.isfinite(u1_pred) and np.isfinite(u2_pred)): actual_len=i; break
        try:
            F_pred = F(t_curr, u1_pred, u2_pred);
            if not np.all(np.isfinite(F_pred)): actual_len=i; break
        except Exception as e: print(f"AM2_M5 F_pred err @ i={i}: {e}"); actual_len=i; break
        u1_new = u1_prev1 + h/12.0*(5.0*F_pred[0] + 8.0*F_prev1[0] - F_prev2[0])
        u2_new = u2_prev1 + h/12.0*(5.0*F_pred[1] + 8.0*F_prev1[1] - F_prev2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        new_z_vec_for_break_am = np.array([u1[i], u2[i]])
        break_occurred_am = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_am[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_am) < 0.5:
                    actual_len = i + 1; break_occurred_am = True
            else:
                if np.linalg.norm(new_z_vec_for_break_am) < 0.001:
                    actual_len = i; break_occurred_am = True
            if break_occurred_am: break
        else:
            if u1[i] <= 0.01: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AM3_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK3_system_M5(F, t_array[:3], u10, u20, v_t_param, v_n_param, d_param)
        start_len=len(u1_start)
        if start_len < 3: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AM3_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(3, len(t_array)):
        t_curr = t_array[i]
        t0, x0, y0 = t_array[i-1], u1[i-1], u2[i-1]
        t1, x1, y1 = t_array[i-2], u1[i-2], u2[i-2]
        t2, x2, y2 = t_array[i-3], u1[i-3], u2[i-3]
        if not all(np.isfinite(v) for v in [x0, y0, x1, y1, x2, y2]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2)
            if not (np.all(np.isfinite(F0)) and np.all(np.isfinite(F1)) and np.all(np.isfinite(F2))): actual_len=i; break
        except Exception as e: print(f"AM3_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_pred = x0 + h/12.0 * (23.0*F0[0] - 16.0*F1[0] + 5.0*F2[0])
        u2_pred = y0 + h/12.0 * (23.0*F0[1] - 16.0*F1[1] + 5.0*F2[1])
        if not (np.isfinite(u1_pred) and np.isfinite(u2_pred)): actual_len=i; break
        try:
            F_pred = F(t_curr, u1_pred, u2_pred);
            if not np.all(np.isfinite(F_pred)): actual_len=i; break
        except Exception as e: print(f"AM3_M5 F_pred err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/24.0*(9.0*F_pred[0] + 19.0*F0[0] - 5.0*F1[0] + F2[0])
        u2_new = y0 + h/24.0*(9.0*F_pred[1] + 19.0*F0[1] - 5.0*F1[1] + F2[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        new_z_vec_for_break_am = np.array([u1[i], u2[i]])
        break_occurred_am = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_am[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_am) < 0.5:
                    actual_len = i + 1; break_occurred_am = True
            else:
                if np.linalg.norm(new_z_vec_for_break_am) < 0.001:
                    actual_len = i; break_occurred_am = True
            if break_occurred_am: break
        else:
            if u1[i] <= 0.01: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

def AM4_system_M5(F, t_array, u10, u20, v_t_param=None, v_n_param=None, d_param=None):
    h = t_array[1] - t_array[0]
    u1 = np.zeros(len(t_array)); u2 = np.zeros(len(t_array)); u1[0],u2[0] = u10,u20
    try:
        u1_start, u2_start = RK4_system_M5(F, t_array[:4], u10, u20, v_t_param, v_n_param, d_param)
        start_len=len(u1_start)
        if start_len < 4: return u1[:start_len], u2[:start_len]
        u1[:start_len], u2[:start_len] = u1_start, u2_start
    except Exception as e: print(f"AM4_M5 init err: {e}"); return u1[:1],u2[:1]
    actual_len=len(t_array)
    for i in range(4, len(t_array)):
        t_curr=t_array[i]
        t0,x0,y0=t_array[i-1],u1[i-1],u2[i-1]; t1,x1,y1=t_array[i-2],u1[i-2],u2[i-2]
        t2,x2,y2=t_array[i-3],u1[i-3],u2[i-3]; t3,x3,y3=t_array[i-4],u1[i-4],u2[i-4]
        if not all(np.isfinite(v) for v in [x0,y0,x1,y1,x2,y2,x3,y3]): actual_len=i; break
        try:
            F0=F(t0,x0,y0); F1=F(t1,x1,y1); F2=F(t2,x2,y2); F3=F(t3,x3,y3)
            if not all(np.all(np.isfinite(f)) for f in [F0, F1, F2, F3]): actual_len=i; break
        except Exception as e: print(f"AM4_M5 F err @ i={i}: {e}"); actual_len=i; break
        u1_pred = x0 + h/24.0 * (55.0*F0[0] - 59.0*F1[0] + 37.0*F2[0] - 9.0*F3[0])
        u2_pred = y0 + h/24.0 * (55.0*F0[1] - 59.0*F1[1] + 37.0*F2[1] - 9.0*F3[1])
        if not (np.isfinite(u1_pred) and np.isfinite(u2_pred)): actual_len=i; break
        try:
            F_pred = F(t_curr, u1_pred, u2_pred);
            if not np.all(np.isfinite(F_pred)): actual_len=i; break
        except Exception as e: print(f"AM4_M5 F_pred err @ i={i}: {e}"); actual_len=i; break
        u1_new = x0 + h/720.0*(251.0*F_pred[0] + 646.0*F0[0] - 264.0*F1[0] + 106.0*F2[0] - 19.0*F3[0])
        u2_new = y0 + h/720.0*(251.0*F_pred[1] + 646.0*F0[1] - 264.0*F1[1] + 106.0*F2[1] - 19.0*F3[1])
        if not (np.isfinite(u1_new) and np.isfinite(u2_new)): actual_len=i; break
        u1[i], u2[i] = u1_new, u2_new
        new_z_vec_for_break_am = np.array([u1[i], u2[i]])
        break_occurred_am = False
        if v_t_param is not None and v_n_param is not None and d_param is not None:
            if v_n_param == 2 * v_t_param or v_t_param == 2 * v_n_param:
                if new_z_vec_for_break_am[0] < -0.1 * d_param or np.linalg.norm(new_z_vec_for_break_am) < 0.5:
                    actual_len = i + 1; break_occurred_am = True
            else:
                if np.linalg.norm(new_z_vec_for_break_am) < 0.001:
                    actual_len = i; break_occurred_am = True
            if break_occurred_am: break
        else:
            if u1[i] <= 0.01: actual_len = i + 1; break
    return u1[:actual_len], u2[:actual_len]

# =================================================================
#  NEW SOLVERS for Model 5, Simulation 2 (Combined Logic)
# =================================================================
# --- RK Methods (Initializer for AB/AM) ---
def RK2_system_M5_Sim2_CombinedLogic(f_combined_ode_func, t_array_initial_segment, initial_state_combined, catch_dist_threshold):
    h_step = t_array_initial_segment[1] - t_array_initial_segment[0]
    time_points_list = [t_array_initial_segment[0]]
    state_history_list = [initial_state_combined]
    caught_flag_main = False
    time_of_catch_main = t_array_initial_segment[-1]
    t_curr, current_st = t_array_initial_segment[0], initial_state_combined
    try:
        k1 = f_combined_ode_func(t_curr, current_st)
        k2 = f_combined_ode_func(t_curr + h_step / 2.0, current_st + h_step / 2.0 * k1)
        k2_heun = f_combined_ode_func(t_curr + h_step, current_st + h_step * k1)
        new_st = current_st + h_step / 2.0 * (k1 + k2_heun)
    except Exception as e:
        print(f"RK2_Sim2_Combined: Error in f_combined_ode_func call: {e}")
        return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main
    time_points_list.append(t_array_initial_segment[1])
    state_history_list.append(new_st)
    if np.linalg.norm(new_st[2:4] - new_st[0:2]) <= catch_dist_threshold:
        caught_flag_main = True
        time_of_catch_main = t_array_initial_segment[1]
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def RK3_system_M5_Sim2_CombinedLogic(f_combined_ode_func, t_array_initial_segment, initial_state_combined, catch_dist_threshold):
    h_step = t_array_initial_segment[1] - t_array_initial_segment[0]
    time_points_list = [t_array_initial_segment[0]]
    state_history_list = [initial_state_combined]
    caught_flag_main = False
    time_of_catch_main = t_array_initial_segment[-1]
    current_st_loop = initial_state_combined
    for i in range(len(t_array_initial_segment) - 1): 
        t_curr = t_array_initial_segment[i]
        try:
            k1 = f_combined_ode_func(t_curr, current_st_loop)
            k2 = f_combined_ode_func(t_curr + h_step / 2.0, current_st_loop + h_step / 2.0 * k1)
            k3 = f_combined_ode_func(t_curr + h_step, current_st_loop - h_step * k1 + 2.0 * h_step * k2)
            new_st = current_st_loop + h_step / 6.0 * (k1 + 4.0 * k2 + k3)
        except Exception as e:
            print(f"RK3_Sim2_Combined: Error in f_combined_ode_func call at t={t_curr:.2f}: {e}")
            return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main
        time_points_list.append(t_array_initial_segment[i+1])
        state_history_list.append(new_st)
        current_st_loop = new_st 
        if np.linalg.norm(new_st[2:4] - new_st[0:2]) <= catch_dist_threshold:
            caught_flag_main = True
            time_of_catch_main = t_array_initial_segment[i+1]
            break 
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def RK4_system_M5_Sim2_CombinedLogic(f_combined_ode_func, t_array_initial_segment, initial_state_combined, catch_dist_threshold):
    h_step = t_array_initial_segment[1] - t_array_initial_segment[0]
    time_points_list = [t_array_initial_segment[0]]
    state_history_list = [initial_state_combined]
    caught_flag_main = False
    time_of_catch_main = t_array_initial_segment[-1]
    current_st_loop = initial_state_combined
    for i in range(len(t_array_initial_segment) - 1): 
        t_curr = t_array_initial_segment[i]
        try:
            k1 = f_combined_ode_func(t_curr, current_st_loop)
            k2 = f_combined_ode_func(t_curr + h_step / 2.0, current_st_loop + h_step / 2.0 * k1)
            k3 = f_combined_ode_func(t_curr + h_step / 2.0, current_st_loop + h_step / 2.0 * k2)
            k4 = f_combined_ode_func(t_curr + h_step, current_st_loop + h_step * k3)
            new_st = current_st_loop + h_step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        except Exception as e:
            print(f"RK4_Sim2_Combined: Error in f_combined_ode_func call at t={t_curr:.2f}: {e}")
            return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_mai
        time_points_list.append(t_array_initial_segment[i+1])
        state_history_list.append(new_st)
        current_st_loop = new_st
        if np.linalg.norm(new_st[2:4] - new_st[0:2]) <= catch_dist_threshold:
            caught_flag_main = True
            time_of_catch_main = t_array_initial_segment[i+1]
            break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def RK5_system_M5_Sim2_CombinedLogic(f_combined_ode_func, t_array_initial_segment, initial_state_combined, catch_dist_threshold):
    h_step = t_array_initial_segment[1] - t_array_initial_segment[0]
    time_points_list = [t_array_initial_segment[0]]
    state_history_list = [initial_state_combined]
    caught_flag_main = False
    time_of_catch_main = t_array_initial_segment[-1]
    current_st_loop = initial_state_combined
    for i in range(len(t_array_initial_segment) - 1):
        t_curr = t_array_initial_segment[i]
        try:
            k1 = f_combined_ode_func(t_curr, current_st_loop)
            k2 = f_combined_ode_func(t_curr + h_step/4.0, current_st_loop + h_step/4.0 * k1)
            k3 = f_combined_ode_func(t_curr + h_step/4.0, current_st_loop + h_step/8.0 * k1 + h_step/8.0 * k2)
            k4 = f_combined_ode_func(t_curr + h_step/2.0, current_st_loop - h_step/2.0 * k2 + h_step * k3)
            k5 = f_combined_ode_func(t_curr + 3.0*h_step/4.0, current_st_loop + 3.0*h_step/16.0 * k1 + 9.0*h_step/16.0 * k4)
            k6 = f_combined_ode_func(t_curr + h_step, current_st_loop - 3.0*h_step/7.0 * k1 + 2.0*h_step/7.0 * k2 + 12.0*h_step/7.0 * k3 - 12.0*h_step/7.0 * k4 + 8.0*h_step/7.0 * k5)
            new_st = current_st_loop + h_step/90.0 * (7.0*k1 + 32.0*k3 + 12.0*k4 + 32.0*k5 + 7.0*k6)
        except Exception as e:
            print(f"RK5_Sim2_Combined: Error in f_combined_ode_func call at t={t_curr:.2f}: {e}")
            return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main
        time_points_list.append(t_array_initial_segment[i+1])
        state_history_list.append(new_st)
        current_st_loop = new_st
        if np.linalg.norm(new_st[2:4] - new_st[0:2]) <= catch_dist_threshold:
            caught_flag_main = True
            time_of_catch_main = t_array_initial_segment[i+1]
            break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

# --- Adams-Bashforth Methods ---
def AB2_system_M5_Sim2_CombinedLogic(f_combined_like, t_array_full_potential, initial_state_combined, catch_dist_threshold):
    num_initial_points = 2 
    t_init_segment = t_array_full_potential[:num_initial_points]
    time_points_list, states_init, caught_init, time_catch_init = RK2_system_M5_Sim2_CombinedLogic(f_combined_like, t_init_segment, initial_state_combined, catch_dist_threshold)
    if caught_init or len(states_init) < num_initial_points:
        return time_points_list, states_init, caught_init, time_catch_init
    time_points_list = list(time_points_list)
    state_history_list = list(states_init)
    h_step = t_array_full_potential[1] - t_array_full_potential[0]
    caught_flag_main = False
    time_of_catch_main = t_array_full_potential[-1]
    f_values_history = [f_combined_like(time_points_list[j], state_history_list[j]) for j in range(num_initial_points)]
    for i in range(num_initial_points - 1, len(t_array_full_potential) - 1):
        try:
            y_i = state_history_list[i]
            f_i = f_values_history[i]
            f_im1 = f_values_history[i-1]
            y_ip1 = y_i + (h_step / 2.0) * (3.0 * f_i - 1.0 * f_im1)
        except Exception as e:
            print(f"AB2_Sim2_Combined: Error calculating y_ip1 at t={time_points_list[i]:.2f}: {e}")
            break 
        state_history_list.append(y_ip1)
        time_points_list.append(t_array_full_potential[i+1])
        if np.linalg.norm(y_ip1[2:4] - y_ip1[0:2]) <= catch_dist_threshold:
            caught_flag_main = True
            time_of_catch_main = t_array_full_potential[i+1]
            break 
        try:
            f_ip1 = f_combined_like(t_array_full_potential[i+1], y_ip1)
            f_values_history.append(f_ip1)
        except Exception as e:
            print(f"AB2_Sim2_Combined: Error calculating f_ip1 at t={t_array_full_potential[i+1]:.2f}: {e}")
            break

    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def AB3_system_M5_Sim2_CombinedLogic(f_combined_like, t_array_full_potential, initial_state_combined, catch_dist_threshold):
    num_initial_points = 3 
    t_init_segment = t_array_full_potential[:num_initial_points]
    time_points_list, states_init, caught_init, time_catch_init = RK3_system_M5_Sim2_CombinedLogic(f_combined_like, t_init_segment, initial_state_combined, catch_dist_threshold)
    if caught_init or len(states_init) < num_initial_points:
        return time_points_list, states_init, caught_init, time_catch_init
    time_points_list = list(time_points_list)
    state_history_list = list(states_init)
    h_step = t_array_full_potential[1] - t_array_full_potential[0]
    caught_flag_main = False
    time_of_catch_main = t_array_full_potential[-1]
    f_values_history = [f_combined_like(time_points_list[j], state_history_list[j]) for j in range(num_initial_points)]
    for i in range(num_initial_points - 1, len(t_array_full_potential) - 1):
        try:
            y_i = state_history_list[i]
            f_i = f_values_history[i]
            f_im1 = f_values_history[i-1]
            f_im2 = f_values_history[i-2]
            y_ip1 = y_i + (h_step / 12.0) * (23.0 * f_i - 16.0 * f_im1 + 5.0 * f_im2)
        except Exception as e:
            print(f"AB3_Sim2_Combined: Error calculating y_ip1 at t={time_points_list[i]:.2f}: {e}")
            break
        state_history_list.append(y_ip1)
        time_points_list.append(t_array_full_potential[i+1])
        if np.linalg.norm(y_ip1[2:4] - y_ip1[0:2]) <= catch_dist_threshold:
            caught_flag_main = True; time_of_catch_main = t_array_full_potential[i+1]; break
        try:
            f_ip1 = f_combined_like(t_array_full_potential[i+1], y_ip1)
            f_values_history.append(f_ip1)
        except Exception as e:
            print(f"AB3_Sim2_Combined: Error calculating f_ip1 at t={t_array_full_potential[i+1]:.2f}: {e}"); break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def AB4_system_M5_Sim2_CombinedLogic(f_combined_like, t_array_full_potential, initial_state_combined, catch_dist_threshold):
    num_initial_points = 4 
    t_init_segment = t_array_full_potential[:num_initial_points]
    time_points_list, states_init, caught_init, time_catch_init = RK4_system_M5_Sim2_CombinedLogic(f_combined_like, t_init_segment, initial_state_combined, catch_dist_threshold)
    if caught_init or len(states_init) < num_initial_points: return time_points_list, states_init, caught_init, time_catch_init
    time_points_list = list(time_points_list); state_history_list = list(states_init)
    h_step = t_array_full_potential[1] - t_array_full_potential[0]; caught_flag_main = False; time_of_catch_main = t_array_full_potential[-1]
    f_values_history = [f_combined_like(time_points_list[j], state_history_list[j]) for j in range(num_initial_points)]
    for i in range(num_initial_points - 1, len(t_array_full_potential) - 1):
        try:
            y_i = state_history_list[i]; f_i = f_values_history[i]; f_im1 = f_values_history[i-1]; f_im2 = f_values_history[i-2]; f_im3 = f_values_history[i-3]
            y_ip1 = y_i + (h_step / 24.0) * (55.0 * f_i - 59.0 * f_im1 + 37.0 * f_im2 - 9.0 * f_im3)
        except Exception as e: print(f"AB4_Sim2_Combined: Error y_ip1 t={time_points_list[i]:.2f}: {e}"); break
        state_history_list.append(y_ip1); time_points_list.append(t_array_full_potential[i+1])
        if np.linalg.norm(y_ip1[2:4] - y_ip1[0:2]) <= catch_dist_threshold: caught_flag_main = True; time_of_catch_main = t_array_full_potential[i+1]; break
        try: f_ip1 = f_combined_like(t_array_full_potential[i+1], y_ip1); f_values_history.append(f_ip1)
        except Exception as e: print(f"AB4_Sim2_Combined: Error f_ip1 t={t_array_full_potential[i+1]:.2f}: {e}"); break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def AB5_system_M5_Sim2_CombinedLogic(f_combined_like, t_array_full_potential, initial_state_combined, catch_dist_threshold):
    num_initial_points = 5 
    t_init_segment = t_array_full_potential[:num_initial_points]
    time_points_list, states_init, caught_init, time_catch_init = RK5_system_M5_Sim2_CombinedLogic(f_combined_like, t_init_segment, initial_state_combined, catch_dist_threshold)
    if caught_init or len(states_init) < num_initial_points: return time_points_list, states_init, caught_init, time_catch_init
    time_points_list = list(time_points_list); state_history_list = list(states_init)
    h_step = t_array_full_potential[1] - t_array_full_potential[0]; caught_flag_main = False; time_of_catch_main = t_array_full_potential[-1]
    f_values_history = [f_combined_like(time_points_list[j], state_history_list[j]) for j in range(num_initial_points)]
    for i in range(num_initial_points - 1, len(t_array_full_potential) - 1):
        try:
            y_i = state_history_list[i]; f_i = f_values_history[i]; f_im1 = f_values_history[i-1]; f_im2 = f_values_history[i-2]; f_im3 = f_values_history[i-3]; f_im4 = f_values_history[i-4]
            y_ip1 = y_i + (h_step / 720.0) * (1901.0 * f_i - 2774.0 * f_im1 + 2616.0 * f_im2 - 1274.0 * f_im3 + 251.0 * f_im4)
        except Exception as e: print(f"AB5_Sim2_Combined: Error y_ip1 t={time_points_list[i]:.2f}: {e}"); break
        state_history_list.append(y_ip1); time_points_list.append(t_array_full_potential[i+1])
        if np.linalg.norm(y_ip1[2:4] - y_ip1[0:2]) <= catch_dist_threshold: caught_flag_main = True; time_of_catch_main = t_array_full_potential[i+1]; break
        try: f_ip1 = f_combined_like(t_array_full_potential[i+1], y_ip1); f_values_history.append(f_ip1)
        except Exception as e: print(f"AB5_Sim2_Combined: Error f_ip1 t={t_array_full_potential[i+1]:.2f}: {e}"); break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

# --- Adams-Moulton Methods ---
def AM2_system_M5_Sim2_CombinedLogic(f_combined_like, t_array_full_potential, initial_state_combined, catch_dist_threshold):
    num_initial_points = 2 
    t_init_segment = t_array_full_potential[:num_initial_points]
    time_points_list, states_init, caught_init, time_catch_init = RK2_system_M5_Sim2_CombinedLogic(f_combined_like, t_init_segment, initial_state_combined, catch_dist_threshold)
    if caught_init or len(states_init) < num_initial_points: return time_points_list, states_init, caught_init, time_catch_init
    time_points_list = list(time_points_list); state_history_list = list(states_init)
    h_step = t_array_full_potential[1] - t_array_full_potential[0]; caught_flag_main = False; time_of_catch_main = t_array_full_potential[-1]
    f_values_history = [f_combined_like(time_points_list[j], state_history_list[j]) for j in range(num_initial_points)] 
    for i in range(num_initial_points - 1, len(t_array_full_potential) - 1): 
        try:
            y_i = state_history_list[i]; t_i = time_points_list[i]
            f_i = f_values_history[i]
            f_im1 = f_values_history[i-1] 
            y_pred_ip1 = y_i + (h_step / 2.0) * (3.0 * f_i - 1.0 * f_im1)
            t_ip1 = t_array_full_potential[i+1]
            f_pred_ip1 = f_combined_like(t_ip1, y_pred_ip1) 
            y_ip1 = y_i + (h_step / 12.0) * (5.0 * f_pred_ip1 + 8.0 * f_i - 1.0 * f_im1)
        except Exception as e: print(f"AM2_Sim2_Combined: Error y_ip1 t={t_i:.2f}: {e}"); break
        state_history_list.append(y_ip1); time_points_list.append(t_ip1)
        if np.linalg.norm(y_ip1[2:4] - y_ip1[0:2]) <= catch_dist_threshold: caught_flag_main = True; time_of_catch_main = t_ip1; break
        try: f_ip1_corrected = f_combined_like(t_ip1, y_ip1); f_values_history.append(f_ip1_corrected)
        except Exception as e: print(f"AM2_Sim2_Combined: Error f_ip1_corr t={t_ip1:.2f}: {e}"); break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def AM3_system_M5_Sim2_CombinedLogic(f_combined_like, t_array_full_potential, initial_state_combined, catch_dist_threshold):
    num_initial_points = 3 
    t_init_segment = t_array_full_potential[:num_initial_points]
    time_points_list, states_init, caught_init, time_catch_init = RK3_system_M5_Sim2_CombinedLogic(f_combined_like, t_init_segment, initial_state_combined, catch_dist_threshold)
    if caught_init or len(states_init) < num_initial_points: return time_points_list, states_init, caught_init, time_catch_init
    time_points_list = list(time_points_list); state_history_list = list(states_init)
    h_step = t_array_full_potential[1] - t_array_full_potential[0]; caught_flag_main = False; time_of_catch_main = t_array_full_potential[-1]
    f_values_history = [f_combined_like(time_points_list[j], state_history_list[j]) for j in range(num_initial_points)]
    for i in range(num_initial_points - 1, len(t_array_full_potential) - 1):
        try:
            y_i = state_history_list[i]; t_i = time_points_list[i]
            f_i = f_values_history[i]; f_im1 = f_values_history[i-1]; f_im2 = f_values_history[i-2]
            y_pred_ip1 = y_i + (h_step / 12.0) * (23.0 * f_i - 16.0 * f_im1 + 5.0 * f_im2)
            t_ip1 = t_array_full_potential[i+1]
            f_pred_ip1 = f_combined_like(t_ip1, y_pred_ip1)
            y_ip1 = y_i + (h_step / 24.0) * (9.0 * f_pred_ip1 + 19.0 * f_i - 5.0 * f_im1 + 1.0 * f_im2)
        except Exception as e: print(f"AM3_Sim2_Combined: Error y_ip1 t={t_i:.2f}: {e}"); break
        state_history_list.append(y_ip1); time_points_list.append(t_ip1)
        if np.linalg.norm(y_ip1[2:4] - y_ip1[0:2]) <= catch_dist_threshold: caught_flag_main = True; time_of_catch_main = t_ip1; break
        try: f_ip1_corrected = f_combined_like(t_ip1, y_ip1); f_values_history.append(f_ip1_corrected)
        except Exception as e: print(f"AM3_Sim2_Combined: Error f_ip1_corr t={t_ip1:.2f}: {e}"); break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main

def AM4_system_M5_Sim2_CombinedLogic(f_combined_like, t_array_full_potential, initial_state_combined, catch_dist_threshold):
    num_initial_points = 4 
    t_init_segment = t_array_full_potential[:num_initial_points]
    time_points_list, states_init, caught_init, time_catch_init = RK4_system_M5_Sim2_CombinedLogic(f_combined_like, t_init_segment, initial_state_combined, catch_dist_threshold)
    if caught_init or len(states_init) < num_initial_points: return time_points_list, states_init, caught_init, time_catch_init
    time_points_list = list(time_points_list); state_history_list = list(states_init)
    h_step = t_array_full_potential[1] - t_array_full_potential[0]; caught_flag_main = False; time_of_catch_main = t_array_full_potential[-1]
    f_values_history = [f_combined_like(time_points_list[j], state_history_list[j]) for j in range(num_initial_points)]
    for i in range(num_initial_points - 1, len(t_array_full_potential) - 1):
        try:
            y_i = state_history_list[i]; t_i = time_points_list[i]
            f_i = f_values_history[i]; f_im1 = f_values_history[i-1]; f_im2 = f_values_history[i-2]; f_im3 = f_values_history[i-3]
            y_pred_ip1 = y_i + (h_step / 24.0) * (55.0 * f_i - 59.0 * f_im1 + 37.0 * f_im2 - 9.0 * f_im3)
            t_ip1 = t_array_full_potential[i+1]
            f_pred_ip1 = f_combined_like(t_ip1, y_pred_ip1)
            y_ip1 = y_i + (h_step / 720.0) * (251.0 * f_pred_ip1 + 646.0 * f_i - 264.0 * f_im1 + 106.0 * f_im2 - 19.0 * f_im3)
        except Exception as e: print(f"AM4_Sim2_Combined: Error y_ip1 t={t_i:.2f}: {e}"); break
        state_history_list.append(y_ip1); time_points_list.append(t_ip1)
        if np.linalg.norm(y_ip1[2:4] - y_ip1[0:2]) <= catch_dist_threshold: caught_flag_main = True; time_of_catch_main = t_ip1; break
        try: f_ip1_corrected = f_combined_like(t_ip1, y_ip1); f_values_history.append(f_ip1_corrected)
        except Exception as e: print(f"AM4_Sim2_Combined: Error f_ip1_corr t={t_ip1:.2f}: {e}"); break
    return np.array(time_points_list), np.array(state_history_list), caught_flag_main, time_of_catch_main
# ==============================================
#           ODE Function Definitions
# ==============================================
# Các hàm này sẽ được tham chiếu bởi MODELS_DATA để tránh lỗi serialization

def get_model1_ode(k):
    return lambda t, y: k * y

def get_model1_exact(O0, k, t0): # SỬA: O₀ -> O0, t₀ -> t0
    return lambda t: O0 * np.exp(k * (np.asarray(t) - t0))

def get_model2_ode(c):
    return lambda t, y: c * (y**(2.0/3.0) + 1e-15)

def get_model2_exact(x0, c, t0): # SỬA: x₀ -> x0, t₀ -> t0
    return lambda t: (x0**(1.0/3.0) + c * (np.asarray(t) - t0) / 3.0)**3

def get_model3_ode(r, n_initial):
    return lambda t, y: -r * y * (n_initial + 1.0 - y)

def get_model3_exact(n_initial, r, t0): # SỬA: t₀ -> t0
    if n_initial <= 0:
        return lambda t: np.zeros_like(np.asarray(t))
    return lambda t: (n_initial * (n_initial + 1.0) * np.exp(-r * (n_initial + 1.0) * (np.asarray(t) - t0))) / \
                     (1.0 + n_initial * np.exp(-r * (n_initial + 1.0) * (np.asarray(t) - t0)))

def get_model4_ode(alpha, beta, m, G, l):
    return lambda t, u1, u2: np.array([u2, m * l * G - alpha * u2 - beta * u1])

def get_model4_exact(alpha, beta, m, G, l, n, k, t0): # SỬA: t₀ -> t0
    return lambda t_arr: _model4_exact_solution(alpha, beta, m, G, l, n, k, t0, t_arr)

def get_model5_ode(u_param, v_param):
    return lambda t, x, y: _model5_ode_system(t, x, y, u_param, v_param)

# ==============================================
#           Models Data
# ==============================================
AGENT_DIRECTION_CHANGE_PROB = 0.02
AGENT_MAX_ANGLE_PERTURBATION_DEG = 10
MAX_TOTAL_AGENTS_FOR_FULL_DISPLAY = 150 
SAMPLE_SIZE_FOR_LARGE_POPULATION = 100 
ABM_ROOM_DIMENSION_DEFAULT = 10.0
ABM_AGENT_SPEED_DEFAULT = 0.05
ABM_CONTACT_RADIUS_DEFAULT = 0.55
ABM_R_FACTOR_DEFAULT = 1000
ABM_PTRANS_MIN = 0.01
ABM_PTRANS_MAX = 0.9
ABM_MAX_STEPS_DEFAULT = 400
ABM_INTERVAL_DEFAULT = 120

class DiseaseSimulationABM:
    def __init__(self, total_population, initial_infected_count_for_abm, room_dimension,
                 contact_radius, transmission_prob, agent_speed):
        self.n_total_population_initial = total_population
        self.n_total_population = total_population
        self.n_infected_initial_abm = initial_infected_count_for_abm
        self.room_dimension = room_dimension
        self.contact_radius = contact_radius
        self.transmission_prob = transmission_prob
        self.agent_speed = agent_speed
        self.susceptible_coords = np.zeros((0,2))
        self.infected_coords = np.zeros((0,2))
        self.susceptible_velocities = np.zeros((0,2))
        self.infected_velocities = np.zeros((0,2))
        self.current_time_step = 0
        self.contact_radius_patches = []
        self._initialize_agents()

    def _initialize_agents(self):
        if self.n_infected_initial_abm > self.n_total_population:
            raise ValueError("Số nhiễm ban đầu không thể lớn hơn tổng dân số.")
        all_coords = np.random.rand(self.n_total_population, 2) * self.room_dimension
        actual_initial_infected = min(self.n_infected_initial_abm, self.n_total_population)
        if actual_initial_infected > self.n_total_population:
            actual_initial_infected = self.n_total_population      
        if self.n_total_population > 0 and actual_initial_infected > 0 :
             infected_indices = np.random.choice(self.n_total_population, actual_initial_infected, replace=False)
        elif actual_initial_infected == 0:
             infected_indices = np.array([], dtype=int)
        else: 
             infected_indices = np.array([], dtype=int)
        susceptible_mask = np.ones(self.n_total_population, dtype=bool)
        if len(infected_indices) > 0:
            susceptible_mask[infected_indices] = False       
        self.infected_coords = all_coords[infected_indices]
        self.susceptible_coords = all_coords[susceptible_mask]
        num_susceptible = self.susceptible_coords.shape[0]
        if num_susceptible > 0:
            angles_s = np.random.rand(num_susceptible) * 2 * np.pi
            self.susceptible_velocities = self.agent_speed * np.array([np.cos(angles_s), np.sin(angles_s)]).T
        else:
            self.susceptible_velocities = np.zeros((0,2))
        num_infected = self.infected_coords.shape[0]
        if num_infected > 0:
            angles_i = np.random.rand(num_infected) * 2 * np.pi
            self.infected_velocities = self.agent_speed * np.array([np.cos(angles_i), np.sin(angles_i)]).T
        else:
            self.infected_velocities = np.zeros((0,2))

    def _move_agents(self, coords_array, velocities_array):
        if coords_array.shape[0] == 0: return coords_array, velocities_array
        coords_array += velocities_array
        for i in range(coords_array.shape[0]):
            for dim in range(2):
                if coords_array[i, dim] < 0:
                    coords_array[i, dim] = 0
                    velocities_array[i, dim] *= -1
                elif coords_array[i, dim] > self.room_dimension:
                    coords_array[i, dim] = self.room_dimension
                    velocities_array[i, dim] *= -1
        for i in range(velocities_array.shape[0]):
            if np.random.rand() < AGENT_DIRECTION_CHANGE_PROB:
                angle_perturbation_rad = np.deg2rad((np.random.rand() - 0.5) * 2 * AGENT_MAX_ANGLE_PERTURBATION_DEG)
                vx, vy = velocities_array[i, 0], velocities_array[i, 1]
                current_speed_sq = vx**2 + vy**2
                if current_speed_sq < 1e-12 :
                    new_angle = np.random.rand() * 2 * np.pi; speed_to_use = self.agent_speed
                else:
                    current_angle = np.arctan2(vy, vx); new_angle = current_angle + angle_perturbation_rad; speed_to_use = self.agent_speed
                velocities_array[i, 0] = speed_to_use * np.cos(new_angle)
                velocities_array[i, 1] = speed_to_use * np.sin(new_angle)
        return coords_array, velocities_array

    def _check_infections(self):
        if self.susceptible_coords.shape[0] == 0 or self.infected_coords.shape[0] == 0: return
        newly_infected_indices = []
        for i, s_pos in enumerate(self.susceptible_coords):
            if self.infected_coords.shape[0] > 0:
                distances_sq = np.sum((self.infected_coords - s_pos)**2, axis=1)
                min_dist_sq_to_infected = np.min(distances_sq)
                if min_dist_sq_to_infected < self.contact_radius**2:
                    if np.random.rand() < self.transmission_prob: newly_infected_indices.append(i)
        if newly_infected_indices:
            newly_infected_indices_np = np.array(sorted(list(set(newly_infected_indices)), reverse=True))
            agents_coords_to_move = self.susceptible_coords[newly_infected_indices_np]
            agents_velocities_to_move = self.susceptible_velocities[newly_infected_indices_np]
            if agents_coords_to_move.shape[0] > 0:
                 self.infected_coords = np.vstack((self.infected_coords, agents_coords_to_move))
                 self.infected_velocities = np.vstack((self.infected_velocities, agents_velocities_to_move))
            self.susceptible_coords = np.delete(self.susceptible_coords, newly_infected_indices_np, axis=0)
            self.susceptible_velocities = np.delete(self.susceptible_velocities, newly_infected_indices_np, axis=0)

    def step(self):
        self.current_time_step += 1
        self.susceptible_coords, self.susceptible_velocities = self._move_agents(self.susceptible_coords, self.susceptible_velocities)
        self.infected_coords, self.infected_velocities = self._move_agents(self.infected_coords, self.infected_velocities)
        self._check_infections()
        num_susceptible = len(self.susceptible_coords)
        if num_susceptible == 0 and self.n_total_population_initial > 0:
            return True # Simulation ended
        return False
    
    def get_display_coords(self, max_total, sample_size):
        total = self.susceptible_coords.shape[0] + self.infected_coords.shape[0]
        if total <= max_total:
            return self.susceptible_coords, self.infected_coords
        else:
            s_ratio = self.susceptible_coords.shape[0] / total if total > 0 else 0
            s_sample_size = int(sample_size * s_ratio)
            i_sample_size = sample_size - s_sample_size
            
            s_indices = np.random.choice(self.susceptible_coords.shape[0], min(s_sample_size, self.susceptible_coords.shape[0]), replace=False)
            i_indices = np.random.choice(self.infected_coords.shape[0], min(i_sample_size, self.infected_coords.shape[0]), replace=False)
            
            return self.susceptible_coords[s_indices], self.infected_coords[i_indices]

    def get_current_stats(self):
        return {
            "time_step": self.current_time_step,
            "susceptible_count": len(self.susceptible_coords),
            "infected_count": len(self.infected_coords),
            "total_population": self.n_total_population_initial
        }
def _model6_get_ode_func(k1, k2):
    """Tạo hàm f(t, ya, yb, yc) cho Model 6."""
    def f(t, ya, yb, yc):
        da = -k1 * ya
        db = k1 * ya - k2 * yb
        dc = k2 * yb
        return da, db, dc
    return f

def _model6_get_exact_func(k1, k2, yA0, yB0, yC0, t0): # SỬA t₀ thành t0 ở đây
    """Tạo hàm trả về nghiệm giải tích (a(t), b(t), c(t)) cho Model 6."""
    def exact_solution_at_t(t_arr):
        t = np.asarray(t_arr) - t0 # Và ở đây
        
        yA = yA0 * np.exp(-k1 * t)
        
        if abs(k1 - k2) < 1e-15:
            yB = (k1 * yA0 * t + yB0) * np.exp(-k1 * t)
        else:
            term1 = (k1 * yA0) / (k2 - k1)
            term2 = np.exp(-k1 * t) - np.exp(-k2 * t)
            yB = term1 * term2 + yB0 * np.exp(-k2 * t)

        total_initial = yA0 + yB0 + yC0
        yC = total_initial - yA - yB
        
        return yA, yB, yC
    return exact_solution_at_t
	
MODELS_DATA = {
    LANG_VI["model1_name"]: {
        "id": "model1",
        "equation_key": "model1_eq",
        "description_key": "model1_desc",
        "param_keys_vi": [LANG_VI["model1_param1"], LANG_VI["model1_param2"], LANG_VI["model1_param3"], LANG_VI["model1_param4"]],
        "param_keys_en": [LANG_EN["model1_param1"], LANG_EN["model1_param2"], LANG_EN["model1_param3"], LANG_EN["model1_param4"]],
        "internal_param_keys": ["O0", "k", "t0", "t1"], # SỬA: O₀ -> O0, t₀ -> t0, t₁ -> t1
        "ode_func": get_model1_ode,
        "exact_func": get_model1_exact,
    },
    LANG_VI["model2_name"]: {
        "id": "model2",
        "equation_key": "model2_eq",
        "description_key": "model2_desc",
        "param_keys_vi": [LANG_VI["model2_param1"], LANG_VI["model2_param3"], LANG_VI["model2_param4"]],
        "param_keys_en": [LANG_EN["model2_param1"], LANG_EN["model2_param3"], LANG_EN["model2_param4"]],
        "internal_param_keys": ["x0", "t0", "t1"], # SỬA: x₀ -> x0, t₀ -> t0, t₁ -> t1
        "ode_func": get_model2_ode,
        "exact_func": get_model2_exact,
    },
    LANG_VI["model3_name"]: {
        "id": "model3",
        "can_run_abm_on_screen3": True,
        "equation_key": "model3_eq",
        "description_key": "model3_desc",
        "param_keys_vi": [LANG_VI["model3_param2"], LANG_VI["model3_param4"], LANG_VI["model3_param5"]],
        "param_keys_en": [LANG_EN["model3_param2"], LANG_EN["model3_param4"], LANG_EN["model3_param5"]],
        "internal_param_keys": ["n", "t0", "t1"], # SỬA: t₀ -> t0, t₁ -> t1
        "ode_func": get_model3_ode,
        "exact_func": get_model3_exact,
        "abm_defaults": {
            "initial_infected": 1, "room_dimension": ABM_ROOM_DIMENSION_DEFAULT,
            "r_to_ptrans_factor": 5000, "ptrans_min": ABM_PTRANS_MIN,
            "ptrans_max": ABM_PTRANS_MAX, "base_agent_speed": 0.04,
            "speed_scaling_factor": 0.5, "min_agent_speed": 0.02,
            "max_agent_speed": 0.20, "base_contact_radius": 0.5,
            "radius_scaling_factor": 3.0, "min_contact_radius": 0.3,
            "max_contact_radius": 1.5, "seconds_per_step": 0.1,
            "max_steps": ABM_MAX_STEPS_DEFAULT, "interval_ms": ABM_INTERVAL_DEFAULT,
            "display_max_total": MAX_TOTAL_AGENTS_FOR_FULL_DISPLAY,
            "display_sample_size": SAMPLE_SIZE_FOR_LARGE_POPULATION
        }
    },
    LANG_VI["model4_name"]: {
        "id": "model4", "is_system": True,
        "equation_key": "model4_eq", "description_key": "model4_desc",
        "internal_param_keys": ["m", "l", "a", "s", "G", "Y0", "dY0", "t0", "t1"], # SỬA: t₀ -> t0, t₁ -> t1
        "ode_func": get_model4_ode,
        "exact_func": get_model4_exact,
    },
    LANG_VI["model5_name"]: {
        "id": "model5", "is_system": True, "uses_rk5_reference": True,
        "equation_key": "model5_eq", "description_key": "model5_desc",
        "ode_label_key": "model6_ode_system_label",
        "hide_exact_solution_display": True,
        "internal_param_keys": ["x0", "y0", "u", "v", "t0", "t1"], # SỬA: t₀ -> t0, t₁ -> t1
        "ode_func": get_model5_ode,
        "exact_func": None,
    },
    LANG_VI["model6_name"]: {
        "id": "model6",
        "is_system": True,
        "uses_rk5_reference": False,
        "equation_key": "model6_eq",
        "description_key": "model6_desc",
        "ode_label_key": "model6_ode_system_label",
        "hide_exact_solution_display": False,
        
        "param_keys_vi": [
            LANG_VI["model6_param_yA0"], LANG_VI["model6_param_yB0"], LANG_VI["model6_param_yC0"],
            LANG_VI["model6_param_k1"], LANG_VI["model6_param_k2"],
            LANG_VI["model6_param_t0"], LANG_VI["model6_param_t1"],
        ],
        "param_keys_en": [
            LANG_EN["model6_param_yA0"], LANG_EN["model6_param_yB0"], LANG_EN["model6_param_yC0"],
            LANG_EN["model6_param_k1"], LANG_EN["model6_param_k2"],
            LANG_EN["model6_param_t0"], LANG_EN["model6_param_t1"],
        ],
        "internal_param_keys": ["y_A0", "y_B0", "y_C0", "k1", "k2", "t0", "t1"],
        "ode_func": _model6_get_ode_func,
        "exact_func": _model6_get_exact_func,
        "components": {
            "A": "model6_component_A",
            "B": "model6_component_B",
            "C": "model6_component_C"
        }
    },
}
#Solve model 4
def _model4_exact_solution(alpha, beta, m, G, l, n, k, t0, t_arr):
    t_rel = np.asarray(t_arr) - t0 
    Y_vals = np.zeros_like(t_rel)
    dY_vals = np.zeros_like(t_rel)
    if abs(beta) < 1e-15:
        if abs(alpha) < 1e-15: 
             c = m * l * G
             Y_vals = n + k * t_rel + 0.5 * c * t_rel**2
             dY_vals = k + c * t_rel
             return Y_vals, dY_vals
        else:
             c = m * l * G
             B = (c / alpha - k) / alpha
             A = n - B
             Y_vals = A + B * np.exp(-alpha * t_rel) + (c / alpha) * t_rel
             dY_vals = -alpha * B * np.exp(-alpha * t_rel) + (c / alpha)
             return Y_vals, dY_vals
    # Case beta != 0
    steady_state = (m * l * G) / beta
    delta = alpha**2 - 4 * beta
    if delta > 1e-15: 
        r1 = (-alpha + np.sqrt(delta)) / 2.0
        r2 = (-alpha - np.sqrt(delta)) / 2.0
        if abs(r1 - r2) > 1e-15:
            C2 = (k - r1 * (n - steady_state)) / (r2 - r1)
            C1 = (n - steady_state) - C2
        else:
             C1, C2 = 0, 0 
        Y_vals = C1 * np.exp(r1 * t_rel) + C2 * np.exp(r2 * t_rel) + steady_state
        dY_vals = C1 * r1 * np.exp(r1 * t_rel) + C2 * r2 * np.exp(r2 * t_rel)
    elif delta < -1e-15: # Underdamped
        omega = np.sqrt(-delta) / 2.0
        zeta = -alpha / 2.0
        C1 = n - steady_state
        if abs(omega)>1e-15:
            C2 = (k - zeta * C1) / omega
        else:
            C2 = 0 
        exp_term = np.exp(zeta * t_rel)
        cos_term = np.cos(omega * t_rel)
        sin_term = np.sin(omega * t_rel)
        Y_vals = exp_term * (C1 * cos_term + C2 * sin_term) + steady_state
        dY_vals = zeta * exp_term * (C1 * cos_term + C2 * sin_term) + \
                  exp_term * (-C1 * omega * sin_term + C2 * omega * cos_term)
    else: 
        r = -alpha / 2.0
        C1 = n - steady_state
        C2 = k - r * C1
        Y_vals = (C1 + C2 * t_rel) * np.exp(r * t_rel) + steady_state
        dY_vals = C2 * np.exp(r * t_rel) + (C1 + C2 * t_rel) * r * np.exp(r * t_rel)
    return Y_vals, dY_vals
#Solve model 5
def _model5_ode_system(t, x, y, u, v):
    r = np.sqrt(x**2 + y**2) + 1e-15
    dxdt = -v * x / r
    dydt = -v * y / r - u
    return np.array([dxdt, dydt])

def _model6_rk2(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    for i in range(n - 1):
        da1, db1, dc1 = f(t[i], a[i], b[i], c[i])
        da2, db2, dc2 = f(t[i] + h, a[i] + h * da1, b[i] + h * db1, c[i] + h * dc1)
        a[i + 1] = a[i] + h / 2 * (da1 + da2)
        b[i + 1] = b[i] + h / 2 * (db1 + db2)
        c[i + 1] = c[i] + h / 2 * (dc1 + dc2)
    return a, b, c

def _model6_rk3(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    for i in range(n - 1):
        da1, db1, dc1 = f(t[i], a[i], b[i], c[i])
        da2, db2, dc2 = f(t[i] + h / 2, a[i] + h / 2 * da1, b[i] + h / 2 * db1, c[i] + h / 2 * dc1)
        da3, db3, dc3 = f(t[i] + h, a[i] + h * (-da1 + 2 * da2), b[i] + h * (-db1 + 2 * db2), c[i] + h * (-dc1 + 2 * dc2))
        a[i + 1] = a[i] + h / 6 * (da1 + 4 * da2 + da3)
        b[i + 1] = b[i] + h / 6 * (db1 + 4 * db2 + db3)
        c[i + 1] = c[i] + h / 6 * (dc1 + 4 * dc2 + dc3)
    return a, b, c

def _model6_rk4(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    for i in range(n - 1):
        da1, db1, dc1 = f(t[i], a[i], b[i], c[i])
        da2, db2, dc2 = f(t[i] + h / 2, a[i] + h / 2 * da1, b[i] + h / 2 * db1, c[i] + h / 2 * dc1)
        da3, db3, dc3 = f(t[i] + h / 2, a[i] + h / 2 * da2, b[i] + h / 2 * db2, c[i] + h / 2 * dc2)
        da4, db4, dc4 = f(t[i] + h, a[i] + h * da3, b[i] + h * db3, c[i] + h * dc3)
        a[i + 1] = a[i] + h / 6 * (da1 + 2 * da2 + 2 * da3 + da4)
        b[i + 1] = b[i] + h / 6 * (db1 + 2 * db2 + 2 * db3 + db4)
        c[i + 1] = c[i] + h / 6 * (dc1 + 2 * dc2 + 2 * dc3 + dc4)
    return a, b, c

def _model6_ab2(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    if n > 1:
        exact_a, exact_b, exact_c = exact_sol_func(t[1])
        a[1], b[1], c[1] = exact_a, exact_b, exact_c
    for i in range(1, n - 1):
        da1, db1, dc1 = f(t[i], a[i], b[i], c[i])
        da0, db0, dc0 = f(t[i - 1], a[i - 1], b[i - 1], c[i - 1])
        a[i + 1] = a[i] + h / 2 * (3 * da1 - da0)
        b[i + 1] = b[i] + h / 2 * (3 * db1 - db0)
        c[i + 1] = c[i] + h / 2 * (3 * dc1 - dc0)
    return a, b, c

def _model6_ab3(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    if n > 2:
        exact_a, exact_b, exact_c = exact_sol_func(t[1:3])
        a[1:3], b[1:3], c[1:3] = exact_a, exact_b, exact_c
    elif n > 1:
        exact_a, exact_b, exact_c = exact_sol_func(t[1])
        a[1], b[1], c[1] = exact_a, exact_b, exact_c
    for i in range(2, n - 1):
        da2, db2, dc2 = f(t[i], a[i], b[i], c[i])
        da1, db1, dc1 = f(t[i - 1], a[i - 1], b[i - 1], c[i - 1])
        da0, db0, dc0 = f(t[i - 2], a[i - 2], b[i - 2], c[i - 2])
        a[i + 1] = a[i] + h / 12 * (23 * da2 - 16 * da1 + 5 * da0)
        b[i + 1] = b[i] + h / 12 * (23 * db2 - 16 * db1 + 5 * db0)
        c[i + 1] = c[i] + h / 12 * (23 * dc2 - 16 * dc1 + 5 * dc0)
    return a, b, c

def _model6_ab4(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    if n > 3:
        exact_a, exact_b, exact_c = exact_sol_func(t[1:4])
        a[1:4], b[1:4], c[1:4] = exact_a, exact_b, exact_c
    elif n > 1:
        num_pts = n - 1
        exact_a, exact_b, exact_c = exact_sol_func(t[1:1+num_pts])
        a[1:1+num_pts], b[1:1+num_pts], c[1:1+num_pts] = exact_a, exact_b, exact_c
    for i in range(3, n - 1):
        da3, db3, dc3 = f(t[i], a[i], b[i], c[i])
        da2, db2, dc2 = f(t[i - 1], a[i - 1], b[i - 1], c[i - 1])
        da1, db1, dc1 = f(t[i - 2], a[i - 2], b[i - 2], c[i - 2])
        da0, db0, dc0 = f(t[i - 3], a[i - 3], b[i - 3], c[i - 3])
        a[i + 1] = a[i] + h / 24 * (55 * da3 - 59 * da2 + 37 * da1 - 9 * da0)
        b[i + 1] = b[i] + h / 24 * (55 * db3 - 59 * db2 + 37 * db1 - 9 * db0)
        c[i + 1] = c[i] + h / 24 * (55 * dc3 - 59 * dc2 + 37 * dc1 - 9 * dc0)
    return a, b, c

def _model6_ab5(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    if n > 4:
        exact_a, exact_b, exact_c = exact_sol_func(t[1:5])
        a[1:5], b[1:5], c[1:5] = exact_a, exact_b, exact_c
    elif n > 1:
        num_pts = n - 1
        exact_a, exact_b, exact_c = exact_sol_func(t[1:1+num_pts])
        a[1:1+num_pts], b[1:1+num_pts], c[1:1+num_pts] = exact_a, exact_b, exact_c
    for i in range(4, n - 1):
        da4, db4, dc4 = f(t[i], a[i], b[i], c[i])
        da3, db3, dc3 = f(t[i - 1], a[i - 1], b[i - 1], c[i - 1])
        da2, db2, dc2 = f(t[i - 2], a[i - 2], b[i - 2], c[i - 2])
        da1, db1, dc1 = f(t[i - 3], a[i - 3], b[i - 3], c[i - 3])
        da0, db0, dc0 = f(t[i - 4], a[i - 4], b[i - 4], c[i - 4])
        a[i + 1] = a[i] + h / 720 * (1901 * da4 - 2774 * da3 + 2616 * da2 - 1274 * da1 + 251 * da0)
        b[i + 1] = b[i] + h / 720 * (1901 * db4 - 2774 * db3 + 2616 * db2 - 1274 * db1 + 251 * db0)
        c[i + 1] = c[i] + h / 720 * (1901 * dc4 - 2774 * dc3 + 2616 * dc2 - 1274 * dc1 + 251 * dc0)
    return a, b, c

def _model6_am2(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    if n > 1:
        exact_a, exact_b, exact_c = exact_sol_func(t[1])
        a[1], b[1], c[1] = exact_a, exact_b, exact_c
    else: return a,b,c
    for i in range(1, n - 1):
        da1, db1, dc1 = f(t[i], a[i], b[i], c[i])
        da0, db0, dc0 = f(t[i - 1], a[i - 1], b[i - 1], c[i - 1])
        a_pred = a[i] + h / 2 * (3 * da1 - da0)
        b_pred = b[i] + h / 2 * (3 * db1 - db0)
        c_pred = c[i] + h / 2 * (3 * dc1 - dc0)
        da_pred, db_pred, dc_pred = f(t[i + 1], a_pred, b_pred, c_pred)
        a[i + 1] = a[i] + h / 12 * (5 * da_pred + 8 * da1 - da0)
        b[i + 1] = b[i] + h / 12 * (5 * db_pred + 8 * db1 - db0)
        c[i + 1] = c[i] + h / 12 * (5 * dc_pred + 8 * dc1 - dc0)
    return a, b, c

def _model6_am3(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    if n > 2:
        exact_a, exact_b, exact_c = exact_sol_func(t[1:3])
        a[1:3], b[1:3], c[1:3] = exact_a, exact_b, exact_c
    elif n > 1:
        exact_a, exact_b, exact_c = exact_sol_func(t[1])
        a[1], b[1], c[1] = exact_a, exact_b, exact_c
    else: return a,b,c
    for i in range(2, n - 1):
        da2, db2, dc2 = f(t[i], a[i], b[i], c[i])
        da1, db1, dc1 = f(t[i - 1], a[i - 1], b[i - 1], c[i - 1])
        da0, db0, dc0 = f(t[i - 2], a[i - 2], b[i - 2], c[i - 2])
        a_pred = a[i] + h / 12 * (23 * da2 - 16 * da1 + 5 * da0)
        b_pred = b[i] + h / 12 * (23 * db2 - 16 * db1 + 5 * db0)
        c_pred = c[i] + h / 12 * (23 * dc2 - 16 * dc1 + 5 * dc0)
        da_pred, db_pred, dc_pred = f(t[i + 1], a_pred, b_pred, c_pred)
        a[i + 1] = a[i] + h / 24 * (9 * da_pred + 19 * da2 - 5 * da1 + da0)
        b[i + 1] = b[i] + h / 24 * (9 * db_pred + 19 * db2 - 5 * db1 + db0)
        c[i + 1] = c[i] + h / 24 * (9 * dc_pred + 19 * dc2 - 5 * dc1 + dc0)
    return a, b, c

def _model6_am4(f, exact_sol_func, t, y_A0, y_B0, y_C0):
    h = t[1] - t[0]; n = len(t)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    a[0], b[0], c[0] = y_A0, y_B0, y_C0
    if n > 3:
        exact_a, exact_b, exact_c = exact_sol_func(t[1:4])
        a[1:4], b[1:4], c[1:4] = exact_a, exact_b, exact_c
    elif n > 1:
        num_pts = n - 1
        exact_a, exact_b, exact_c = exact_sol_func(t[1:1+num_pts])
        a[1:1+num_pts], b[1:1+num_pts], c[1:1+num_pts] = exact_a, exact_b, exact_c
    else: return a,b,c
    for i in range(3, n - 1):
        da3, db3, dc3 = f(t[i], a[i], b[i], c[i])
        da2, db2, dc2 = f(t[i - 1], a[i - 1], b[i - 1], c[i - 1])
        da1, db1, dc1 = f(t[i - 2], a[i - 2], b[i - 2], c[i - 2])
        da0, db0, dc0 = f(t[i - 3], a[i - 3], b[i - 3], c[i - 3])
        a_pred = a[i] + h / 24 * (55 * da3 - 59 * da2 + 37 * da1 - 9 * da0)
        b_pred = b[i] + h / 24 * (55 * db3 - 59 * db2 + 37 * db1 - 9 * db0)
        c_pred = c[i] + h / 24 * (55 * dc3 - 59 * dc2 + 37 * dc1 - 9 * dc0)
        da_pred, db_pred, dc_pred = f(t[i + 1], a_pred, b_pred, c_pred)
        a[i + 1] = a[i] + h / 720 * (251 * da_pred + 646 * da3 - 264 * da2 + 106 * da1 - 19 * da0)
        b[i + 1] = b[i] + h / 720 * (251 * db_pred + 646 * db3 - 264 * db2 + 106 * db1 - 19 * db0)
        c[i + 1] = c[i] + h / 720 * (251 * dc_pred + 646 * dc3 - 264 * dc2 + 106 * dc1 - 19 * dc0)
    return a, b, c
# --- 4. CẤU TRÚC CHÍNH CỦA ỨNG DỤNG STREAMLIT ---

# Khởi tạo st.session_state để lưu trạng thái
def initialize_session_state():
    query_params = st.query_params

    # 1. Xác định ngôn ngữ (ưu tiên query param)
    query_lang = query_params.get("lang")
    if query_lang and query_lang in ['vi', 'en']:
        st.session_state.lang = query_lang
    elif 'lang' not in st.session_state:
        st.session_state.lang = 'vi'

    # 2. Tải file ngôn ngữ
    st.session_state.translations = load_language_file(st.session_state.lang)

    # 3. Xác định trang (ưu tiên query param)
    query_page = query_params.get("page")
    if query_page and query_page in ['welcome', 'model_selection', 'simulation', 'dynamic_simulation']:
        st.session_state.page = query_page
    elif 'page' not in st.session_state:
        st.session_state.page = 'welcome'

    # 4. Xác định trang con của welcome (ưu tiên query param)
    query_subpage = query_params.get("subpage")
    if query_subpage:
        st.session_state.welcome_subpage = query_subpage
    elif 'welcome_subpage' not in st.session_state:
        st.session_state.welcome_subpage = "home"
    
    # Khởi tạo các biến khác
    if 'selected_model_key' not in st.session_state: st.session_state.selected_model_key = None
    if 'simulation_results' not in st.session_state: st.session_state.simulation_results = {}
    if 'validated_params' not in st.session_state: st.session_state.validated_params = {}
    if 'anim_running' not in st.session_state: st.session_state.anim_running = False
    if 'anim_frame' not in st.session_state: st.session_state.anim_frame = 0
    if 'm5_scenario' not in st.session_state: st.session_state.m5_scenario = 1

		
# Hàm tiện ích để dịch văn bản


# Hàm chính để điều hướng giữa các trang
def main():
    # Bước 1: Khởi tạo session_state
    initialize_session_state()

    # Bước 2: Cấu hình trang
    # Sử dụng tên file icon chính xác của bạn
    icon_path = os.path.join(FIG_FOLDER, "icon-app-eureka.ico") 
    
    page_icon_to_use = icon_path 

    st.set_page_config(
        layout="wide", 
        page_title=tr("app_title"),
        page_icon=page_icon_to_use
    )
    render_navbar() 
    # Bước 3: Chạy logic điều hướng trang
    if st.session_state.page == 'welcome':
        show_welcome_page()
    elif st.session_state.page == 'model_selection':
        show_model_selection_page()
    elif st.session_state.page == 'simulation':
        show_simulation_page()
    elif st.session_state.page == 'dynamic_simulation':
        show_dynamic_simulation_page()

# Các hàm render trang (sẽ được định nghĩa ở các phần sau)
# ==============================================
#           PHẦN 2: TRANG CHÀO MỪNG VÀ CHỌN MÔ HÌNH
# ==============================================

# --- Thay thế hàm show_welcome_page cũ ---
def show_welcome_page():
    # Phần CSS giữ nguyên
    st.markdown("""
        <style>
        .main { background-color: #E6ECF4; }
        div[data-testid="stAppViewBlockContainer"] { padding-top: 2rem; }
        .header-col h2 { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; line-height: 1.4; margin: 0; text-align: center; }
        .project-title { font-size: 3rem; font-weight: bold; color: #1E3A8A; line-height: 1.3; }
        .welcome-text { color: #475569; font-size: 2rem; line-height: 1.6;}
        .welcome-credits h3 { font-size: 1.2rem; font-weight: bold; color: #1E3A8A; }
        .welcome-credits p { font-size: 1rem; color: #334155; margin-bottom: 0; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Chỉ hiển thị nội dung nếu là trang "home"
    if st.session_state.welcome_subpage == "home":
            
            # === KHỐI LOGO VÀ TÊN TRƯỜNG ĐÃ BỊ XÓA ===
            # col1, col2, col3 ... đã bị xóa
            # st.divider() đã bị xóa
            
            # Giữ lại phần giới thiệu chính
            col4, col5 = st.columns([1.5, 1], vertical_alignment="center")
            with col4:
                st.markdown(f"<div class='project-title'>{tr('welcome_project_title').replace('\\n', '<br>')}</div>", unsafe_allow_html=True)
                st.markdown(f"<p class='welcome-text'>{tr('main_desc')}</p>", unsafe_allow_html=True)
            with col5:
                main_image_path = os.path.join(FIG_FOLDER, "multi1.png") 
                if os.path.exists(main_image_path): st.image(main_image_path)
                else: st.warning("Không tìm thấy file 'multi1.png' trong thư mục 'fig'.")
            
            st.write("")
            st.write("") # Thêm một khoảng trống để thay cho phần tên tác giả
            
            # === KHỐI TÊN TÁC GIẢ VÀ GVHD ĐÃ BỊ XÓA ===
            # col6, col7 ... đã bị xóa
            
            # Giữ lại nút bắt đầu
            _, col_start_btn, _ = st.columns([2, 1, 2])
            with col_start_btn:
                if st.button(f"**{tr('start_button')}**", use_container_width=True, type="primary"):
                    st.session_state.page = 'model_selection'
                    st.rerun()

    # Phần trang liên hệ (contact) giữ nguyên
    elif st.session_state.welcome_subpage == "contact":
            lang_code = st.session_state.lang
            contact_filename = f"contact_{lang_code}.txt"
            contact_file_path = os.path.join(base_path, contact_filename)

            if os.path.exists(contact_file_path):
                with open(contact_file_path, "r", encoding="utf-8") as f:
                    contact_content = f.read()
                st.markdown(contact_content, unsafe_allow_html=True)
            else:
                st.error(f"Không tìm thấy file {contact_filename}.")

    st.markdown('</div>', unsafe_allow_html=True)
# --- Thay thế hàm show_model_selection_page cũ ---
def show_model_selection_page():   
    # --- CSS TÙY CHỈNH (giữ nguyên) ---
    st.markdown("""
        <style>
        .main { background-color: #E6ECF4; }
        div[data-testid="stAppViewBlockContainer"] { padding-top: 2rem; }
        .st-emotion-cache-1r4qj8v { margin-bottom: 1rem; } 
        </style>
    """, unsafe_allow_html=True)

    # --- NỘI DUNG CHÍNH CỦA TRANG (logic callback giữ nguyên) ---
    st.title(tr('screen1_title'))
    
    model_display_names = [tr(f"{data['id']}_name") for data in MODELS_DATA.values()]
    model_vi_keys = list(MODELS_DATA.keys())

    def on_model_change():
        selected_display = st.session_state.model_selector
        selected_index = model_display_names.index(selected_display)
        st.session_state.selected_model_key = model_vi_keys[selected_index]

    if 'selected_model_key' not in st.session_state or st.session_state.selected_model_key not in model_vi_keys:
        st.session_state.selected_model_key = model_vi_keys[0]

    current_selection_index = model_vi_keys.index(st.session_state.selected_model_key)
    
    st.selectbox(
        label=" ", options=model_display_names, index=current_selection_index,
        key='model_selector', on_change=on_model_change
    )
    
    model_data = MODELS_DATA[st.session_state.selected_model_key]
    st.write("") 
    
    # --- PHẦN THÔNG TIN MÔ HÌNH (BỐ CỤC XUỐNG DÒNG) ---
    with st.container(border=True):
        st.subheader(tr('screen1_model_info_group_title'))
        
        col_equation, col_description = st.columns([1.2, 1.5])

        # === CỘT BÊN TRÁI: HIỂN THỊ PHƯƠNG TRÌNH (BỐ CỤC XUỐNG DÒNG) ===
        with col_equation:
            eq_text = tr(model_data['equation_key'])
            ode_html, exact_html = (eq_text.split('<br>', 1) + [None])[:2] if '<br>' in eq_text else (eq_text, None)

            # Lấy tiêu đề PTVP (mặc định hoặc tùy chỉnh)
            ode_label_key = model_data.get("ode_label_key", "screen1_ode_label")
            
            # Hiển thị PTVP
            st.markdown(f"**{tr(ode_label_key)}**")
            st.latex(html_to_latex(ode_html.strip()))
            
            # Chỉ hiển thị Nghiệm giải tích nếu có và không bị ẩn
            if exact_html and not model_data.get("hide_exact_solution_display", False):
                # Thêm khoảng trống để tách biệt
                st.write("") 
                st.markdown(f"**{tr('screen1_exact_label')}**")
                st.latex(html_to_latex(exact_html.strip()))

        # === CỘT BÊN PHẢI: HIỂN THỊ MÔ TẢ (giữ nguyên) ===
        with col_description:
            st.markdown(f"**{tr('screen1_description_label')}**")
            full_description_html = tr(model_data['description_key'])
            
            if '<br>' in full_description_html:
                parts = full_description_html.split('<br>', 1)
                general_desc = parts[0]
                params_desc_html = parts[1]
                
                st.markdown(general_desc, unsafe_allow_html=True)
                
                param_list = params_desc_html.split('<br>')
                cleaned_params = [p.strip() for p in param_list if p.strip()] 
                params_desc_markdown = "\n".join([f"- {p}" for p in cleaned_params])
                
                st.markdown(params_desc_markdown, unsafe_allow_html=True)
            else:
                st.markdown(full_description_html, unsafe_allow_html=True)
        
    st.write("") 

    # --- PHẦN ỨNG DỤNG VÀ NÚT BẤM (giữ nguyên) ---
    with st.container(border=True):
        st.subheader(tr('screen1_model_application_group_title'))
        model_id = model_data.get("id")
        lang_suffix = "Vie" if st.session_state.lang == 'vi' else "Eng"
        image_filename = f"model_{model_id[5:]}_{lang_suffix}.png"
        image_path = os.path.join(FIG_FOLDER, image_filename)
        if os.path.exists(image_path):
            st.image(image_path)
        else:
            st.warning(f"Không tìm thấy ảnh: {image_filename}")
    
    st.write("")
    st.write("")
    
    _, col_continue_btn, _ = st.columns([2, 1, 2])
    with col_continue_btn:
        if st.button(tr('screen1_continue_button'), type="primary", use_container_width=True):
            st.session_state.page = 'simulation'
            st.rerun()
        

# ==============================================
#           PHẦN 3 (CẬP NHẬT): TRANG MÔ PHỎNG CHÍNH
# ==============================================

# --- CÁC HÀM PHỤ TRỢ CHO VIỆC TÍNH TOÁN (LẤY TỪ CODE GỐC) ---

def _get_middle_t_value(t_r_pairs):
    if not t_r_pairs: return None
    middle_index = len(t_r_pairs) // 2
    return t_r_pairs[middle_index][0]

def _predict_r_for_model3(t_start, t_end, n_initial):
    a = t_start
    b = t_end
    n = n_initial
    print(f"Calculating r (standalone logic): t_start={a}, t_end={b}, n_initial={n}")
    if n <= 0:
        print("Error: Initial value 'n' must be positive to calculate r.")
        return None
    if b <= a:
        print("Error: t_end must be greater than t_start to calculate r.")
        return None
    t_value_list = []
    try:
         b_int_limit = int(b)
         if b_int_limit <= 1:
            print(f"Warning: int(t_end) ({b_int_limit}) <= 1. Cannot generate t_value list.")
            return None
         t_value_list = [(b - float(i)) / float(i) for i in range(1, b_int_limit)]
         print(f"Generated {len(t_value_list)} t_values using standalone range(1, {b_int_limit}) logic.")
    except Exception as e: print(f"Error generating t_value list with b={b}: {e}"); return None
    if not t_value_list: print("Error: t_value list is empty."); return None
    if not t_value_list:
         print("Error: t_value list is empty.")
         return None
    min_t_value = float(b) / 2.0 if b <= 50 else 10.0
    print(f"  Using min_t_value = {min_t_value}")
    predict = []
    min_r_threshold = 1e-8
    for t_val in t_value_list:
        if t_val <= 1e-15: continue
        try:
             current_r = (np.log(n)) / ((n + 1.0) * t_val)
        except ValueError:
             print(f"Warning: Math error calculating r for t_val={t_val}. Skipping.")
             continue
        if min_r_threshold <= current_r and b > t_val >= min_t_value:
            predict.append([t_val, current_r])
    if not predict:
        print("Warning: No valid [t_value, r] pairs found meeting criteria.")
        return None
    print(f"  Predict list (unsorted, filtered) has {len(predict)} elements.")
    middle_t = _get_middle_t_value(predict)
    if middle_t is None:
        print("Error: Could not get middle t_value from sorted list.")
        return None
    print(f"  Middle t_value (based on list order) is: {middle_t:.4f}")
    calculated_r = None
    middle_index_find = len(predict) // 2
    if abs(predict[middle_index_find][0] - middle_t) < 1e-9:
         calculated_r = predict[middle_index_find][1]
    else:
        min_diff = float('inf')
        for t, r_val in predict:
            diff = abs(t - middle_t)
            if diff < min_diff:
                min_diff = diff
                calculated_r = r_val
        print(f"  Note: Used closest t_value match (diff={min_diff:.2e}) for r.")
    if calculated_r is not None:
        print(f"Standalone logic calculated r = {calculated_r:.8g} for middle t_value = {middle_t:.4g}")
    else:
        print("Error: Could not find r corresponding to the middle t_value.")
    return calculated_r

def _prepare_simulation_functions(model_data, input_params, selected_method_short):
    """
    Chuẩn bị các hàm ODE, nghiệm giải tích, và các tham số cần thiết.
    Hàm này không thay đổi st.session_state, chỉ trả về kết quả.
    """
    try:
        ode_gen = model_data.get("ode_func")
        exact_gen = model_data.get("exact_func")
        model_id = model_data.get("id")

        if not callable(ode_gen):
            raise ValueError(tr("msg_model_no_ode").format(tr(f"{model_id}_name")))
        
        t_start = input_params['t0']
        t_end = input_params['t1']
        y0 = None
        calculated_params = {} # Dictionary để lưu các giá trị tính toán được
        if model_id == "model6":
            if 'y_A0' not in input_params or 'y_B0' not in input_params or 'y_C0' not in input_params:
                raise ValueError(tr("msg_missing_y0"))
            y0 = [input_params['y_A0'], input_params['y_B0'], input_params['y_C0']]
        elif model_id == "model4":
            if 'Y0' not in input_params or 'dY0' not in input_params: raise ValueError(tr("msg_missing_y0"))
            y0 = [input_params['Y0'], input_params['dY0']]
        elif model_id == "model5":
            if 'x0' not in input_params or 'y0' not in input_params: raise ValueError(tr("msg_missing_y0"))
            y0 = [input_params['x0'], input_params['y0']]
        else:
            y0_key_map = {'model1': 'O0', 'model2': 'x0', 'model3': 'n'}
            y0_key = y0_key_map.get(model_id)
            if y0_key is None or y0_key not in input_params: raise ValueError(tr("msg_missing_y0"))
            y0 = input_params[y0_key]
        
        ode_func = None
        exact_callable = None
        
        if model_id == "model1":
            k = input_params['k']
            ode_func = ode_gen(k)
            if callable(exact_gen): exact_callable = exact_gen(y0, k, t_start)
        
        elif model_id == "model2":
            number_of_double_times = 5.0 if selected_method_short == "Bashforth" else 2.0
            denominator_b = input_params['t1']
            if denominator_b <= 1e-9 or input_params['x0'] < 0: raise ValueError("t1 > 0 và x0 >= 0")
            x0_cbrt_safe = (input_params['x0'] + 1e-15)**(1.0/3.0)
            doubling_factor_N_cbrt = (2.0**number_of_double_times)**(1.0/3.0)
            lower_c = 3.0 * (doubling_factor_N_cbrt - 1.0) * x0_cbrt_safe / denominator_b
            doubling_factor_Nplus1_cbrt = (2.0**(number_of_double_times + 1.0))**(1.0/3.0)
            upper_c = 3.0 * (doubling_factor_Nplus1_cbrt - 1.0) * x0_cbrt_safe / denominator_b
            calculated_c = random.uniform(min(lower_c, upper_c), max(lower_c, upper_c))
            calculated_params['c'] = calculated_c
            ode_func = ode_gen(calculated_c)
            if callable(exact_gen): exact_callable = exact_gen(y0, calculated_c, t_start)
            
        elif model_id == "model3":
            n_initial = y0
            calculated_r = _predict_r_for_model3(t_start, t_end, n_initial)
            if calculated_r is None: raise ValueError("Không thể tự động tính toán tham số 'r'.")
            calculated_params['r'] = calculated_r
            ode_func = ode_gen(calculated_r, n_initial)
            if callable(exact_gen): exact_callable = exact_gen(n_initial, calculated_r, t_start)
            
        elif model_id == "model4":
            m, l, a, s, G = input_params['m'], input_params['l'], input_params['a'], input_params['s'], input_params['G']
            Y0_val, dY0_val = y0[0], y0[1]
            alpha_calculated = m + l * s - l * m * a
            beta_calculated = l * m * s
            calculated_params['alpha'] = alpha_calculated
            calculated_params['beta'] = beta_calculated
            ode_func = ode_gen(alpha_calculated, beta_calculated, m, G, l)
            if callable(exact_gen): exact_callable = exact_gen(alpha_calculated, beta_calculated, m, G, l, Y0_val, dY0_val, t_start)
            
        elif model_id == "model5":
            u_param, v_param = input_params['u'], input_params['v']
            ode_func = ode_gen(u_param, v_param)
            exact_callable = None
        elif model_id == "model6":
            k1, k2 = input_params['k1'], input_params['k2']
            yA0, yB0, yC0 = y0[0], y0[1], y0[2]
            ode_func = ode_gen(k1, k2)
            if callable(exact_gen):
                exact_callable = exact_gen(k1, k2, yA0, yB0, yC0, t_start)
        if not callable(ode_func): raise RuntimeError(f"Lỗi: Hàm ODE không được tạo cho model '{model_id}'.")
        
        return True, (ode_func, exact_callable, y0, t_start, t_end), calculated_params

    except Exception as e:
        st.error(tr("msg_unknown_error_prep").format(e))
        return False, None, {}

# Highlight: Hàm này không cần thay đổi gì nhiều, nó đã hoạt động tốt.
# Ta sẽ giữ nguyên hàm _perform_single_simulation từ code bạn gửi.
def _perform_single_simulation(model_data, ode_func, exact_sol_func, y0, t_start, t_end, method_short, steps_int, h_target, selected_component='x'):
    """
    Thực hiện một lần mô phỏng đầy đủ, bao gồm cả tính toán bậc hội tụ.
    Phiên bản cập nhật để xử lý Model 6.
    """
    is_system = model_data.get("is_system", False)
    uses_rk5_reference = model_data.get("uses_rk5_reference", False)
    model_id = model_data.get("id")

    # --- Chọn hàm solver ---
    method_map_model6 = {
        "Bashforth": {2: _model6_ab2, 3: _model6_ab3, 4: _model6_ab4, 5: _model6_ab5},
        "Moulton": {2: _model6_am2, 3: _model6_am3, 4: _model6_am4},
        "RungeKutta": {2: _model6_rk2, 3: _model6_rk3, 4: _model6_rk4} # THÊM RK CHO MODEL 6
    }
    method_map_single = {
        "Bashforth": {2: AB2, 3: AB3, 4: AB4, 5: AB5}, 
        "Moulton": {2: AM2, 3: AM3, 4: AM4},
        "RungeKutta": {2: RK2, 3: RK3, 4: RK4} # THÊM RK CHO PT ĐƠN
    }
    method_map_system = {
        "Bashforth": {2: AB2_system, 3: AB3_system, 4: AB4_system, 5: AB5_system}, 
        "Moulton": {2: AM2_system, 3: AM3_system, 4: AM4_system},
        "RungeKutta": {2: RK2_system, 3: RK3_system, 4: RK4_system} # THÊM RK CHO HỆ PT
    }
    method_map_model5 = {
        "Bashforth": {2: AB2_original_system_M5, 3: AB3_original_system_M5, 4: AB4_original_system_M5, 5: AB5_original_system_M5},
        "Moulton": {2: AM2_original_system_M5, 3: AM3_original_system_M5, 4: AM4_original_system_M5},
        "RungeKutta": {2: RK2_original_system_M5, 3: RK3_original_system_M5, 4: RK4_original_system_M5} # THÊM RK CHO MODEL 5
    }
    
    current_map = method_map_single
    if model_id == "model6": current_map = method_map_model6
    elif model_id == "model5": current_map = method_map_model5
    elif is_system: current_map = method_map_system
    
    method_func = current_map.get(method_short, {}).get(steps_int)
    if method_func is None:
        st.error(f"Solver không tồn tại cho {method_short} {steps_int} bước/bậc.")
        return None

    # --- Logic tính toán cho đồ thị và bậc hội tụ ---
    interval_length = t_end - t_start
    if interval_length <= 1e-9: return None
    min_n_required = max(steps_int, 2)
    n_plot = max(int(np.ceil(interval_length / h_target)), min_n_required if model_id != "model5" else 5)
    if uses_rk5_reference: n_plot = max(n_plot, 50)
    h_actual_plot = interval_length / n_plot
    t_plot = np.linspace(t_start, t_end, n_plot + 1)
    
    y_approx_plot_u1, y_exact_plot_u1 = None, None
    y_approx_plot_all_components, y_exact_plot_all_components = None, None
    
    # THÊM NHÁNH XỬ LÝ RIÊNG CHO MODEL 6
    try:
        if model_id == "model6":
            # Gọi solver đặc biệt với 3 điều kiện đầu
            y_approx_plot_all_components = list(method_func(ode_func, exact_sol_func, t_plot, y0[0], y0[1], y0[2]))
            if exact_sol_func:
                y_exact_plot_all_components = list(exact_sol_func(t_plot))
            
            # Chọn thành phần để vẽ dựa trên selected_component
            comp_map_idx = {'A': 0, 'B': 1, 'C': 2}
            selected_idx = comp_map_idx.get(selected_component, 0)
            y_approx_plot_u1 = y_approx_plot_all_components[selected_idx]
            if y_exact_plot_all_components:
                y_exact_plot_u1 = y_exact_plot_all_components[selected_idx]

        elif is_system:
            u1_plot, u2_plot = method_func(ode_func, t_plot, y0[0], y0[1])
            y_approx_plot_all_components = [np.asarray(u1_plot), np.asarray(u2_plot)]
            y_approx_plot_u1 = y_approx_plot_all_components[0 if selected_component == 'x' else 1]

            if uses_rk5_reference:
                rk5_ref_for_screen2 = RK5_original_system_M5
                rk5_ref_x_plot, rk5_ref_y_plot = rk5_ref_for_screen2(ode_func, t_plot, y0[0], y0[1])
                y_exact_plot_all_components = [np.asarray(rk5_ref_x_plot), np.asarray(rk5_ref_y_plot)]
                y_exact_plot_u1 = y_exact_plot_all_components[0 if selected_component == 'x' else 1]
            elif exact_sol_func:
                exact_tuple = exact_sol_func(t_plot)
                if exact_tuple is not None and len(exact_tuple) >= 2:
                    y_exact_plot_all_components = [np.asarray(exact_tuple[0]), np.asarray(exact_tuple[1])]
                    y_exact_plot_u1 = y_exact_plot_all_components[0]
        else: # Phương trình đơn
            y_plot = method_func(ode_func, t_plot, y0)
            y_approx_plot_u1 = np.asarray(y_plot)
            if exact_sol_func:
                y_exact_plot_u1 = np.asarray(exact_sol_func(t_plot))
        
        # Khớp độ dài mảng (quan trọng)
        if y_approx_plot_u1 is not None:
            min_len_plot = len(y_approx_plot_u1)
            if y_exact_plot_u1 is not None: min_len_plot = min(min_len_plot, len(y_exact_plot_u1))
            min_len_plot = min(min_len_plot, len(t_plot))
            t_plot = t_plot[:min_len_plot]
            y_approx_plot_u1 = y_approx_plot_u1[:min_len_plot]
            if y_exact_plot_u1 is not None: y_exact_plot_u1 = y_exact_plot_u1[:min_len_plot]
            if y_approx_plot_all_components: y_approx_plot_all_components = [comp[:min_len_plot] for comp in y_approx_plot_all_components]
            if y_exact_plot_all_components: y_exact_plot_all_components = [comp[:min_len_plot] for comp in y_exact_plot_all_components]

    except Exception as e_plot_approx:
        print(f"Error calculating APPROXIMATE solution for main plot: {e_plot_approx}")
        return None

    # --- Vòng lặp tính bậc hội tụ ---
    errors_convergence, h_values_for_loglog_list, n_values_plotted = [], [], []
    log_h_conv, log_err_conv = [], []
    slope = np.nan
    
    # Logic tạo N values (giữ nguyên)
    # ...
    if model_id == "model6": # Thêm điều kiện cho Model 6
        n_values_for_conv_loop = np.arange(5, 16, 1, dtype=int)
    else:
        # ... (logic cũ cho các model khác)
        interval_n_base = max(1, int(np.ceil(interval_length)))
        if model_id == "model1": n_start_conv, n_end_conv = max(5, 2*interval_n_base), max(20, 10*interval_n_base)
        elif model_id == "model2": n_start_conv, n_end_conv = max(5,interval_n_base), max(10,interval_n_base*2)
        elif model_id == "model3": n_start_conv = interval_n_base; n_end_conv = 3 * interval_n_base
        elif model_id == "model4": n_start_conv, n_end_conv = max(10, 10*interval_n_base), max(30, 30*interval_n_base)
        elif model_id == "model5": n_start_conv, n_end_conv = 2000, 10000
        else: n_start_conv, n_end_conv = 10, 100
        
        if n_start_conv > n_end_conv: n_start_conv, n_end_conv = n_end_conv, n_start_conv
        if n_start_conv <= 0: n_start_conv = max(1, min_n_required)
        if n_end_conv <= n_start_conv: n_end_conv = n_start_conv + 9
        
        num_points_convergence = 10 if model_id != "model5" else 8
        n_values_for_conv_loop = np.linspace(n_start_conv, n_end_conv, num_points_convergence, dtype=int)


    n_values_filtered_conv = np.unique(n_values_for_conv_loop[n_values_for_conv_loop >= min_n_required])
    if len(n_values_filtered_conv) < 2:
        print(f"Warning: Not enough N values for convergence plot.")
    else:
        for n_conv_original in n_values_filtered_conv:
            # ... (logic nhân N_eff giữ nguyên)
            n_eff_conv_sim = n_conv_original
            if model_id == 'model2': n_eff_conv_sim = n_conv_original * 2
            elif model_id == 'model3': n_eff_conv_sim = n_conv_original * 10
            elif model_id == 'model4': n_eff_conv_sim = n_conv_original * 2

            if n_eff_conv_sim <= 0: continue
            h_for_logplot_conv = interval_length / n_eff_conv_sim
            n_eff_conv_sim = max(n_eff_conv_sim, min_n_required)
            t_conv_loop = np.linspace(t_start, t_end, n_eff_conv_sim + 1)
            if len(t_conv_loop) < steps_int + 1: continue

            try:
                y_approx_conv_u1_selected_loop, y_exact_conv_u1_selected_loop = None, None
                
                # THÊM NHÁNH XỬ LÝ RIÊNG CHO MODEL 6 TRONG VÒNG LẶP HỘI TỤ
                if model_id == "model6":
                    approx_all = list(method_func(ode_func, exact_sol_func, t_conv_loop, y0[0], y0[1], y0[2]))
                    exact_all = list(exact_sol_func(t_conv_loop)) if exact_sol_func else [None]*3
                    idx = {'A': 0, 'B': 1, 'C': 2}.get(selected_component, 0)
                    y_approx_conv_u1_selected_loop = approx_all[idx]
                    y_exact_conv_u1_selected_loop = exact_all[idx]
                elif is_system:
                    u1, u2 = method_func(ode_func, t_conv_loop, y0[0], y0[1])
                    y_approx_conv_u1_selected_loop = u1 if selected_component == 'x' else u2
                    
                    if uses_rk5_reference:
                        ex1, ex2 = RK5_original_system_M5(ode_func, t_conv_loop, y0[0], y0[1])
                    elif exact_sol_func:
                        ex1, ex2 = exact_sol_func(t_conv_loop)
                    else: 
                        ex1, ex2 = None, None
                    y_exact_conv_u1_selected_loop = ex1 if selected_component == 'x' else ex2
                else: # Phương trình đơn
                    y_approx_conv_u1_selected_loop = method_func(ode_func, t_conv_loop, y0)
                    if exact_sol_func:
                        y_exact_conv_u1_selected_loop = exact_sol_func(t_conv_loop)
                
                if y_approx_conv_u1_selected_loop is None or y_exact_conv_u1_selected_loop is None : continue
                
                approx_c, exact_c = np.asarray(y_approx_conv_u1_selected_loop), np.asarray(y_exact_conv_u1_selected_loop)
                min_len_c = min(len(approx_c), len(exact_c))
                if min_len_c < 2: continue
                
                error_c = np.linalg.norm(exact_c[:min_len_c] - approx_c[:min_len_c], np.inf)
                if np.isfinite(error_c) and error_c > 1e-16 and h_for_logplot_conv > 1e-16:
                    errors_convergence.append(error_c)
                    n_values_plotted.append(n_conv_original)
                    h_values_for_loglog_list.append(h_for_logplot_conv)
            except Exception as e_conv_loop:
                 print(f"Error during convergence step N_orig={n_conv_original}: {e_conv_loop}")
    
    # ... (logic tính polyfit giữ nguyên)
    if len(errors_convergence) >= 2:
        try:
            log_h_conv = np.log(np.array(h_values_for_loglog_list))
            log_err_conv = np.log(np.array(errors_convergence))
            if len(log_h_conv) >= 2:
                coeffs = np.polyfit(log_h_conv, log_err_conv, 1)
                slope = coeffs[0]
        except Exception as fit_e:
            print(f"Error during polyfit: {fit_e}")
            slope = np.nan

    return {
        "t_plot": np.asarray(t_plot),
        "exact_sol_plot": np.asarray(y_exact_plot_u1) if y_exact_plot_u1 is not None else None,
        "approx_sol_plot": np.asarray(y_approx_plot_u1) if y_approx_plot_u1 is not None else None,
        "exact_sol_plot_all_components": y_exact_plot_all_components,
        "approx_sol_plot_all_components": y_approx_plot_all_components,
        "h_values_for_loglog": np.array(h_values_for_loglog_list),
        "errors_convergence": np.array(errors_convergence),
        "log_h_convergence": log_h_conv,
        "log_error_convergence": log_err_conv,
        "order_slope": slope,
        "n_values_convergence": np.array(n_values_plotted),
        "h_actual_plot": h_actual_plot,
        "n_plot": n_plot,
        "selected_component": selected_component
    }

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
# Highlight: Toàn bộ hàm show_simulation_page được viết lại
def show_simulation_page():
    if not st.session_state.selected_model_key:
        st.warning(tr("msg_select_model_first"))
        if st.button(tr("go_back_to_select"), type="primary"):
            st.session_state.page = 'model_selection'
            st.rerun()
        return

    model_data = MODELS_DATA[st.session_state.selected_model_key]
    model_id = model_data.get("id", "")
    model_name_tr = tr(f"{model_id}_name")
    
    # --- THANH BÊN (SIDEBAR) ---
    with st.sidebar:
        def _cleanup_and_go_to_model_selection():
            st.session_state.simulation_results = {}
            st.session_state.validated_params = {}
            keys_to_clear = [k for k in st.session_state if k.startswith('last_calculated_')]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.page = 'model_selection'
            st.rerun()

        # SỬA LỖI 2: Thay use_container_width
        if st.button(tr("screen2_back_button"), width='stretch', type="secondary"):
            _cleanup_and_go_to_model_selection()
        
        st.title(tr("sidebar_title"))
        st.header(tr('screen2_method_group'))
        select_ab = st.checkbox(tr('screen2_method_ab'), value=True, key='cb_ab')
        select_am = st.checkbox(tr('screen2_method_am'), value=False, key='cb_am')
        select_rk = st.checkbox(tr('screen2_method_rk'), value=False, key='cb_rk')
        st.divider()
        with st.form(key='simulation_form'):
            st.header(tr('screen2_details_group'))
            # 1. Khu vực chi tiết cho Adams-Bashforth
            if select_ab:
                st.subheader(tr('screen2_details_group_ab'))
                step_options_ab = {tr('screen2_step2'): 2, tr('screen2_step3'): 3, tr('screen2_step4'): 4}
                # AB hỗ trợ 5 bước (trừ model 5)
                if model_id != "model5":
                    step_options_ab[tr('screen2_step5')] = 5
                
                if st.checkbox(tr('screen2_select_all_steps_cb'), value=False, key='cb_all_steps_ab'):
                    default_steps_ab = list(step_options_ab.keys())
                else:
                    default_steps_ab = [tr('screen2_step4')] if tr('screen2_step4') in step_options_ab else [list(step_options_ab.keys())[0]]

                st.multiselect(
                    tr('screen2_steps_label'), 
                    options=list(step_options_ab.keys()), 
                    default=default_steps_ab, 
                    key='ms_steps_ab' # KEY RIÊNG
                )
                st.divider()

            # 2. Khu vực chi tiết cho Adams-Moulton
            if select_am:
                # Dùng subheader khác để phân biệt, có thể thêm vào file ngôn ngữ
                st.subheader(tr('screen2_details_group_am')) 
                # AM chỉ hỗ trợ đến 4 bước
                step_options_am = {tr('screen2_step2'): 2, tr('screen2_step3'): 3, tr('screen2_step4'): 4}
                
                if st.checkbox(tr('screen2_select_all_steps_cb'), value=False, key='cb_all_steps_am'):
                     default_steps_am = list(step_options_am.keys())
                else:
                     default_steps_am = [tr('screen2_step4')]

                st.multiselect(
                    tr('screen2_steps_label'), 
                    options=list(step_options_am.keys()), 
                    default=default_steps_am, 
                    key='ms_steps_am' # KEY RIÊNG
                )
                st.divider()

            # 3. Khu vực chi tiết cho Runge-Kutta
            if select_rk:
                st.subheader(tr('screen2_details_group_rk'))
                order_options = {tr('screen2_order2'): 2, tr('screen2_order3'): 3, tr('screen2_order4'): 4}
                all_order_keys = list(order_options.keys())
                
                if st.checkbox(tr('screen2_select_all_steps_cb'), value=False, key='cb_all_orders_rk'):
                    default_orders = all_order_keys
                else:
                    default_orders = [tr('screen2_order4')]
                
                st.multiselect(
                    tr('screen2_order_label'), 
                    options=all_order_keys, 
                    default=default_orders, 
                    key='ms_orders' # Key này đã riêng biệt
                )
                st.divider()

            h_values = ["0.1", "0.05", "0.01", "0.005", "0.001"]
            selected_h_str = st.radio(tr('screen2_h_label'), options=h_values, index=2, horizontal=True)
            
            st.header(tr('screen2_params_group'))
            param_inputs = {}
            internal_keys = model_data.get("internal_param_keys", [])
            current_defaults = MODEL_DEFAULTS.get(model_id, {})
            
            param_labels_key = f"param_keys_{st.session_state.lang}"
            all_param_labels = model_data.get(param_labels_key, model_data.get("param_keys_vi", []))
            for i, key in enumerate(internal_keys):
                label = all_param_labels[i] if i < len(all_param_labels) else key
                default_val = current_defaults.get(key, 1.0)
                param_inputs[key] = st.number_input(label, value=float(default_val), step=1e-4, format="%.4f", key=f"param_{model_id}_{key}")

            selected_component = 'x'
            if model_id == "model6":
                comp_data_m6 = model_data.get("components", {})
                comp_options_m6_display = [tr(v) for v in comp_data_m6.values()]
                comp_options_m6_keys = list(comp_data_m6.keys())
                selected_comp_disp_m6 = st.radio(tr('model6_select_component'), comp_options_m6_display, horizontal=True, key=f"comp_{model_id}")
                selected_component = comp_options_m6_keys[comp_options_m6_display.index(selected_comp_disp_m6)]
            elif model_id == "model5":
                comp_options_m5 = {tr('model5_component_x'): 'x', tr('model5_component_y'): 'y'}
                selected_comp_disp_m5 = st.radio(tr('model5_select_component'), list(comp_options_m5.keys()), horizontal=True, key=f"comp_{model_id}")
                selected_component = comp_options_m5[selected_comp_disp_m5]
            
            # SỬA LỖI 2: Thay use_container_width
            submitted = st.form_submit_button(tr('screen2_init_button'), type="primary", width='stretch')

        # SỬA LỖI 2: Thay use_container_width
        if st.button(tr('screen2_refresh_button'), width='stretch'):
            st.session_state.simulation_results = {}
            st.session_state.validated_params = {}
            st.rerun()

    # --- KHU VỰC HIỂN THỊ CHÍNH ---
    st.header(tr('simulation_results_title'))
    st.subheader(model_name_tr)

    if submitted:
        # ... (logic xử lý form submit không thay đổi)
        with st.spinner(tr('screen2_info_area_running')):
            step_options = {tr('screen2_step2'): 2, tr('screen2_step3'): 3, tr('screen2_step4'): 4, tr('screen2_step5'): 5}
            order_options = {tr('screen2_order2'): 2, tr('screen2_order3'): 3, tr('screen2_order4'): 4}
            
            tasks_to_run = {}
            is_valid = True
            error_messages = []
            
            if st.session_state.cb_ab:
                if 'ms_steps_ab' in st.session_state and st.session_state.ms_steps_ab:
                    tasks_to_run["Bashforth"] = [step_options[s] for s in st.session_state.ms_steps_ab]
                else:
                    is_valid = False
                    error_messages.append(tr('msg_select_step_for_method', "Vui lòng chọn bước cho {0}.").format(tr('screen2_method_ab')))

            if st.session_state.cb_am:
                if 'ms_steps_am' in st.session_state and st.session_state.ms_steps_am:
                    tasks_to_run["Moulton"] = [step_options[s] for s in st.session_state.ms_steps_am]
                else:
                    is_valid = False
                    error_messages.append(tr('msg_select_step_for_method', "Vui lòng chọn bước cho {0}.").format(tr('screen2_method_am')))

            if st.session_state.cb_rk:
                if 'ms_orders' in st.session_state and st.session_state.ms_orders:
                    tasks_to_run["RungeKutta"] = [order_options[o] for o in st.session_state.ms_orders]
                else:
                    is_valid = False
                    error_messages.append(tr('msg_select_step_for_method', "Vui lòng chọn bậc cho {0}.").format(tr('screen2_method_rk')))
            if not (st.session_state.cb_ab or st.session_state.cb_am or st.session_state.cb_rk):
                error_messages.append(tr('msg_select_method')); is_valid = False
            if 't0' in param_inputs and 't1' in param_inputs and param_inputs['t1'] <= param_inputs['t0']:
                error_messages.append(tr('msg_t_end_error')); is_valid = False
            
            if not is_valid:
                for msg in error_messages: st.toast(msg, icon='⚠️')
            else:
                for key in ['last_calculated_c', 'last_calculated_r', 'last_calculated_alpha', 'last_calculated_beta']:
                    if key in st.session_state: del st.session_state[key]
                first_method_short = next(iter(tasks_to_run), "Bashforth")
                prep_ok, prep_data, calculated_params = _prepare_simulation_functions(model_data, param_inputs, first_method_short)
                if prep_ok:
                    for key, value in calculated_params.items():
                        st.session_state[f'last_calculated_{key}'] = value
                    ode_func, exact_callable, y0, t_start, t_end = prep_data
                    results_dict = {}
                    for method_short, steps_list in tasks_to_run.items():
                        if not steps_list: continue
                        results_dict[method_short] = {}
                        for steps in sorted(list(set(steps_list))):
                            res = _perform_single_simulation(
                                model_data, ode_func, exact_callable, y0, t_start, t_end, 
                                method_short, steps, float(selected_h_str), selected_component
                            )
                            if res: results_dict[method_short][steps] = res
                    results_dict = {m: sr for m, sr in results_dict.items() if sr}
                    st.session_state.simulation_results = results_dict
                    st.session_state.validated_params = {
                        'params': param_inputs, 'h_target': float(selected_h_str), 'model_id': model_id,
                        'selected_component': selected_component, 'tasks_run': tasks_to_run
                    }
                    st.rerun()
                else: st.session_state.simulation_results = {}

    results = st.session_state.get('simulation_results', {})
    if not results:
        st.info(tr('screen2_info_area_init'))
    else:
        validated_params = st.session_state.validated_params
        if 'last_calculated_c' in st.session_state and validated_params.get('model_id') == 'model2':
            st.info(f"**{tr('model2_calculated_c_label')}** {st.session_state.last_calculated_c:.6g}")
        if 'last_calculated_r' in st.session_state and validated_params.get('model_id') == 'model3':
            st.info(f"**{tr('model3_calculated_r_label')}** {st.session_state.last_calculated_r:.8g}")
        if 'last_calculated_alpha' in st.session_state and validated_params.get('model_id') == 'model4':
            col_a, col_b = st.columns(2)
            col_a.info(f"**{tr('model4_param_alpha')}:** {st.session_state.last_calculated_alpha:.6g}")
            col_b.info(f"**{tr('model4_param_beta')}:** {st.session_state.last_calculated_beta:.6g}")

        can_run_dynamic = model_data.get("can_run_abm_on_screen3", False) or model_id in ['model2', 'model5']
        if can_run_dynamic:
            # =============================================================
            # SỬA LỖI 1 & 2 TẠI ĐÂY
            if st.button(
                tr("screen2_goto_screen3_button"), 
				use_container_width=True, 
                type="primary",
				key="goto_dynamic_sim_btn"
            ):
            # =============================================================
                first_method_key = next(iter(results.keys()), None)
                if first_method_key:
                    first_method_results = results[first_method_key]
                    highest_step = max(first_method_results.keys(), key=int) if first_method_results else None
                    if highest_step:
                        st.session_state.validated_params_for_dynamic = validated_params.copy()
                        st.session_state.page = 'dynamic_simulation'
                        st.rerun()
                    else: st.toast("Không có kết quả hợp lệ để tạo mô phỏng động.", icon="⚠️")
                else: st.toast("Không có kết quả hợp lệ để tạo mô phỏng động.", icon="⚠️")
        
        # Phần hiển thị tab và đồ thị không thay đổi
        tab1, tab2, tab3, tab4 = st.tabs([
            f"📊 {tr('screen2_plot_solution_title')}", f"📉 {tr('screen2_plot_error_title')}", 
            f"📈 {tr('screen2_plot_order_title')}", f"🔢 {tr('screen2_show_data_button')}"
        ])
        
        # (Hàm generate_and_get_figures và các tab còn lại giữ nguyên)
        @st.cache_data
        def generate_and_get_figures(results_data_json, lang_code, model_id, component):
            font_path = os.path.join(base_path, "fonts", "DejaVuSans.ttf")
            if os.path.exists(font_path):
                from matplotlib.font_manager import FontProperties
                font_prop = FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
	
            translations = load_language_file(lang_code)
            def _tr(key): return translations.get(key, key)
            results_data = json.loads(results_data_json)
		    
            figs = {}
		    # --- 1. Tạo danh sách tất cả các lần chạy và gán màu ---
            all_runs = []
            for method_short, step_results in results_data.items():
                for step_str, res in step_results.items():
                    if res:
                        label = ""
                        step_or_order = int(step_str)
		                # Sửa logic để thêm RK vào label
                        if method_short == "RungeKutta":
                            label = f"RK{step_or_order}"
                        else:
                            label = f"A{method_short[0]}{step_or_order}"
		
                        all_runs.append({
		                    'method': method_short,
		                    'step_or_order': step_or_order,
		                    'result': res,
		                    'label': label
		                })
		
            if not all_runs: return {'solution': Figure(), 'error': Figure(), 'order': Figure()}
		    
            all_runs.sort(key=lambda x: (x['method'], x['step_or_order']))
            num_runs = len(all_runs)
            colors = plt.cm.turbo(np.linspace(0, 1, num_runs)) if num_runs > 1 else ['#1f77b4']
		
		    # --- 2. Lấy thông tin chung ---
            sol_ylabel_text = _tr('screen2_plot_value_axis')
            if model_id == "model5":
                sol_ylabel_text += f" ({component.upper()})"
            elif model_id == "model6":
		        # Truy cập MODELS_DATA từ global scope để lấy thông tin components
                model6_data = MODELS_DATA.get(LANG_VI["model6_name"], {})
                lang_key = model6_data.get("components", {}).get(component, "")
                sol_ylabel_text = _tr(lang_key)
		
            exact_label_key = "screen2_plot_ref_sol_label" if model_id == "model5" else "screen2_plot_exact_sol_label"
		
		    # --- 3. Vẽ các đồ thị ---
            plot_figsize = (7, 5)
		
		    # Đồ thị nghiệm
            fig_sol = Figure(figsize=plot_figsize)
            ax_sol = fig_sol.subplots()
            exact_plotted = False
            for i, run in enumerate(all_runs):
                res, color, label = run['result'], colors[i], run['label']
                t, ap, ex = res.get('t_plot'), res.get('approx_sol_plot'), res.get('exact_sol_plot')
                if not exact_plotted and ex is not None and t is not None and len(t) > 0:
                    ax_sol.plot(t, ex, color='black', ls='--', label=_tr(exact_label_key))
                    exact_plotted = True
                if t is not None and ap is not None and len(t) > 0:
                    ax_sol.plot(t, ap, color=color, label=label)
		
            ax_sol.set_title(_tr('screen2_plot_solution_title'))
            ax_sol.set_xlabel(_tr('screen2_plot_t_axis'))
            ax_sol.set_ylabel(sol_ylabel_text)
            ax_sol.grid(True, linestyle=':'); ax_sol.legend(fontsize='small')
            fig_sol.tight_layout()
            figs['solution'] = fig_sol
		    
		    # Đồ thị sai số
            fig_err = Figure(figsize=plot_figsize)
            ax_err = fig_err.subplots()
            for i, run in enumerate(all_runs):
                res, color, label = run['result'], colors[i], run['label']
                n_vals, err = res.get('n_values_convergence'), res.get('errors_convergence')
		        # Sửa lỗi: Cần đảm bảo err cũng là một list/array
                if n_vals is not None and err is not None and len(n_vals) > 0 and len(err) > 0:
                    ax_err.plot(n_vals, err, marker='.', ms=3, ls='-', color=color, label=label)
		            
            ax_err.set_title(_tr('screen2_plot_error_title'))
            ax_err.set_xlabel(_tr('screen2_plot_n_axis'))
            ax_err.set_ylabel(_tr('screen2_plot_error_axis'))
            ax_err.set_yscale('log')
            ax_err.grid(True, which='both', linestyle=':'); ax_err.legend(fontsize='small')
            fig_err.tight_layout()
            figs['error'] = fig_err

		    # Đồ thị bậc hội tụ
            fig_ord = Figure(figsize=plot_figsize)
            ax_ord = fig_ord.subplots()
            for i, run in enumerate(all_runs):
                res, color, label = run['result'], colors[i], run['label']
                log_h, log_err = res.get('log_h_convergence'), res.get('log_error_convergence')
                if log_h is not None and log_err is not None and len(log_h) >= 2 and len(log_err) >= 2:
                    slope = res.get('order_slope', 0)
                    # Tạo chuỗi Fit linh hoạt, không phụ thuộc vào key dịch
                    fit_label_text = f"Fit: O(h^{slope:.2f})"
                    # Chuyển đổi sang định dạng LaTeX cho Matplotlib
                    fit_label_mathtext = f"$O(h^{{{slope:.2f}}})$"
                    ax_ord.plot(log_h, log_err, 'o', ms=3, color=color, label=f"{label} {_tr('screen2_plot_order_data_label_suffix')}")
                    if np.isfinite(slope):
                        try:
                            fit_line = np.polyval(np.polyfit(log_h, log_err, 1), log_h)
                            ax_ord.plot(log_h, fit_line, '-', color=color, label=fit_label_mathtext)
                        except Exception as e_polyfit:
                            print(f"Could not perform polyfit for {label}: {e_polyfit}")
		                
            ax_ord.set_title(_tr('screen2_plot_order_title'))
            ax_ord.set_xlabel(_tr('screen2_plot_log_h_axis'))
            ax_ord.set_ylabel(_tr('screen2_plot_log_error_axis'))
            ax_ord.grid(True, linestyle=':'); ax_ord.legend(fontsize='small')
            fig_ord.tight_layout()
            figs['order'] = fig_ord
            return figs

        results_json = json.dumps(results, cls=NumpyEncoder)
        figures = generate_and_get_figures(
            results_json,
            st.session_state.lang, 
            validated_params['model_id'], 
            validated_params.get('selected_component', 'x')
        )
        
        with tab1: st.pyplot(figures['solution'])
        with tab2: st.pyplot(figures['error'])
        with tab3: st.pyplot(figures['order'])
        with tab4:
            method_order = ["Bashforth", "Moulton", "RungeKutta"]
            sorted_methods = sorted(results.keys(), key=lambda x: method_order.index(x) if x in method_order else 99)
            for method_short in sorted_methods:
                step_results = results.get(method_short, {})
                if not step_results: continue
                method_key_map = {"Bashforth": "ab", "Moulton": "am", "RungeKutta": "rk"}
                abbreviation = method_key_map.get(method_short, method_short.lower())
                method_display_name = tr(f'screen2_method_{abbreviation}')
                st.subheader(method_display_name)
                for step_str, res in sorted(step_results.items()):
                    step = int(step_str)
                    if method_short == "RungeKutta":
                        run_title_label = f"{tr('screen2_order_label')} {step}"
                    else:
                        run_title_label = f"{tr('screen2_steps_label')} {step}"
                    with st.expander(label=run_title_label):
                        slope_str = f"{res.get('order_slope', 'N/A'):.4f}" if isinstance(res.get('order_slope'), float) else "N/A"
                        st.markdown(f"**{tr('screen2_info_area_show_data_order')}** {slope_str}")
                        t = res.get('t_plot'); approx = res.get('approx_sol_plot'); exact = res.get('exact_sol_plot')
                        if t is not None and approx is not None and len(t) > 0:
                            # === BẮT ĐẦU THAY ĐỔI ===
                            # 1. Tạo dictionary dữ liệu
                            df_data = {
                                't': t, 
                                tr('screen2_info_area_show_data_approx'): approx
                            }
                            # 2. Tạo dictionary định dạng
                            formatter = {
                                't': '{:.10f}',
                                tr('screen2_info_area_show_data_approx'): '{:.10f}'
                            }
                            # 3. Thêm cột 'Chính xác' và 'Sai số' nếu có
                            if exact is not None:
                                exact_col_name = tr('screen2_info_area_show_data_exact')
                                error_col_name = tr('screen2_info_area_show_data_error')
                                
                                df_data[exact_col_name] = exact
                                df_data[error_col_name] = np.abs(np.array(approx) - np.array(exact))
                                
                                formatter[exact_col_name] = '{:.10f}'
                                # ĐÂY LÀ THAY ĐỔI QUAN TRỌNG NHẤT
                                formatter[error_col_name] = '{:.8e}' # 'e' cho scientific notation
                            
                            # 4. Tạo và hiển thị DataFrame với định dạng mới
                            df = pd.DataFrame(df_data)
                            st.dataframe(df.head(20).style.format(formatter), use_container_width=True, height=300)
                            # === KẾT THÚC THAY ĐỔI ===
                        else:
                            st.write(tr("screen2_info_area_show_data_no_points"))
                st.write("---")
            
# ==============================================
#           PHẦN 4: TRANG MÔ PHỎNG ĐỘNG
# ==============================================
class Cell:
    def __init__(self, x, y, gen=0):
        self.x = x
        self.y = y
        self.gen = gen
        self.last_division = -100

@st.cache_data
def run_and_prepare_m5s1_animation_data(_validated_params_json):
    """
    Chạy lại mô phỏng cho Model 5 Sim 1 để lấy dữ liệu quỹ đạo đầy đủ
    và tính toán các thông số ban đầu cho animation.
    Hàm này được cache để không phải tính lại mỗi khi rerender.
    """
    validated_params = json.loads(_validated_params_json)
    params_s2 = validated_params.get('params', {})
    h_target = validated_params.get('h_target', 0.01)
    
    tasks_run = validated_params.get('tasks_run', {})
    method_short = next(iter(tasks_run.keys()), 'Bashforth')
    method_steps = max(tasks_run.get(method_short, [4]))

    u, v = params_s2.get('u', 0.0), params_s2.get('v', 0.0)
    x0, y0 = params_s2.get('x0', 10.0), params_s2.get('y0', 0.0)
    t0, t1 = params_s2.get('t0', 0.0), params_s2.get('t1', 10.0)

    t_end_final = t1
    if v > u:
        time_to_cross_min = abs(x0 / v) if v != 0 else float('inf')
        t_max_reasonable = time_to_cross_min * 5
        t_end_final = max(t1, t_max_reasonable) + 2.0

    model_data = MODELS_DATA[LANG_VI["model5_name"]]
    ode_func = model_data["ode_func"](u, v)
    
    # Sử dụng bộ solver ..._M5_Sim1 mới
    solver_map_sim1 = {
        "Bashforth": {
                2: AB2_system_M5, 3: AB3_system_M5, 
                4: AB4_system_M5, 5: AB5_system_M5 
            },
            "Moulton": {
                2: AM2_system_M5, 3: AM3_system_M5, 
                4: AM4_system_M5
            }
    }
    solver_func = solver_map_sim1.get(method_short, {}).get(method_steps)
    if not solver_func: solver_func = AB4_system_M5_Sim1 # Fallback an toàn

    num_points = max(200, int(np.ceil((t_end_final - t0) / h_target)))
    t_array_full = np.linspace(t0, t_end_final, num_points + 1)
    
    recalc_x, recalc_y = solver_func(ode_func, t_array_full, x0, y0)
    
    min_len = min(len(recalc_x), len(recalc_y))
    t_array_actual = t_array_full[:min_len]
    z_array_actual = np.column_stack((recalc_x, recalc_y))
    
    # Chuẩn bị dữ liệu đồ họa
    d_val = x0
    x_traj_min, x_traj_max = np.min(z_array_actual[:, 0]), np.max(z_array_actual[:, 0])
    y_traj_min, y_traj_max = np.min(z_array_actual[:, 1]), np.max(z_array_actual[:, 1])

    padding_x_standalone = 0.1 * d_val
    y_range_for_padding_standalone = max(abs(y_traj_max - y_traj_min), 0.5 * d_val)
    padding_y_abs_standalone = 0.2 * y_range_for_padding_standalone

    xlim_min_standalone = min(x_traj_min - padding_x_standalone, -padding_x_standalone)
    xlim_max_standalone = max(x_traj_max + padding_x_standalone, d_val + padding_x_standalone)

    ylim_min_calc = min(y_traj_min - padding_y_abs_standalone, y0 - padding_y_abs_standalone)
    ylim_max_calc = max(y_traj_max + padding_y_abs_standalone, y0 + padding_y_abs_standalone)
    
    # Logic điều chỉnh thêm cho ylim từ PySide6
    if ylim_min_calc >= -1.0:
        ylim_min_standalone = min(ylim_min_calc, -max(2.0, padding_y_abs_standalone))
    else:
        ylim_min_standalone = ylim_min_calc
    
    if ylim_max_calc <= 1.0:
        ylim_max_standalone = max(ylim_max_calc, max(2.0, padding_y_abs_standalone))
    else:
        ylim_max_standalone = ylim_max_calc

    xlim = (xlim_min_standalone, xlim_max_standalone)
    ylim = (ylim_min_standalone, ylim_max_standalone)
    
    arrow_data = None
    if u != 0:
        num_arrows_x, num_arrows_y = 8, 6
        x_coords = np.linspace(0 + d_val / (2 * num_arrows_x), d_val - d_val / (2 * num_arrows_x), num_arrows_x)
        y_coords = np.linspace(ylim[0] + (ylim[1] - ylim[0]) / (2 * num_arrows_y), ylim[1] - (ylim[1] - ylim[0]) / (2 * num_arrows_y), num_arrows_y)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        U_grid = np.zeros_like(X_grid)
        V_grid = -np.sign(u) * np.ones_like(Y_grid)
        arrow_data = (X_grid, Y_grid, U_grid, V_grid)

    return {
        "t_plot": t_array_actual,
        "approx_sol_plot_all_components": [z_array_actual[:, 0], z_array_actual[:, 1]],
        "plot_limits": {'xlim': xlim, 'ylim': ylim},
        "arrow_data": arrow_data
    }
	
def _m5s2_z_tn_base(t, traj_params):
    if not traj_params or not traj_params.get("params_x") or not traj_params.get("params_y"):
        return np.array([0.0, 0.0])
    x_val = traj_params["offset_x"]
    for p in traj_params["params_x"]:
        x_val += p["amp"] * (np.sin(p["freq"] * t + p["phase"]) if p["type"] == 'sin' else np.cos(p["freq"] * t + p["phase"]))
    y_val = traj_params["offset_y"]
    for p in traj_params["params_y"]:
        y_val += p["amp"] * (np.sin(p["freq"] * t + p["phase"]) if p["type"] == 'sin' else np.cos(p["freq"] * t + p["phase"]))
    return np.array([x_val, y_val])

def _m5s2_get_base_submarine_velocity(t, traj_params, v_tn_max):
    vx_base = 0; vy_base = 0
    for p in traj_params["params_x"]: vx_base += p["amp"] * p["freq"] * (np.cos(p["freq"] * t + p["phase"]) if p["type"] == 'sin' else -np.sin(p["freq"] * t + p["phase"]))
    for p in traj_params["params_y"]: vy_base += p["amp"] * p["freq"] * (np.cos(p["freq"] * t + p["phase"]) if p["type"] == 'sin' else -np.sin(p["freq"] * t + p["phase"]))
    v_base_vector = np.array([vx_base, vy_base])
    norm_v_base = np.linalg.norm(v_base_vector)
    if norm_v_base < 1e-9: return np.array([0.0, 0.0])
    return (v_base_vector / norm_v_base) * v_tn_max

def _m5s2_get_smarter_avoidance_info(z_tn, z_kt, v_base_tn_dir, radius_avoid, v_tn_max, strength_avoid, fov_deg):
    vector_tn_to_kt = z_kt - z_tn; distance = np.linalg.norm(vector_tn_to_kt)
    is_avoiding = False; v_avoid_vector = np.array([0.0, 0.0])
    if 0 < distance < radius_avoid:
        dir_tn_to_kt = vector_tn_to_kt / distance
        if np.linalg.norm(v_base_tn_dir) < 1e-6:
            v_avoid_vector = -dir_tn_to_kt * v_tn_max * strength_avoid; is_avoiding = True
        else:
            dot_product = np.dot(v_base_tn_dir, dir_tn_to_kt)
            if dot_product > np.cos(np.deg2rad(fov_deg / 2.0)):
                is_avoiding = True; push_away_dir = -dir_tn_to_kt
                cross_prod_val = v_base_tn_dir[0] * dir_tn_to_kt[1] - v_base_tn_dir[1] * dir_tn_to_kt[0]
                turn_dir = np.array([0.0, 0.0])
                if cross_prod_val > 0.05: turn_dir = np.array([v_base_tn_dir[1], -v_base_tn_dir[0]])
                elif cross_prod_val < -0.05: turn_dir = np.array([-v_base_tn_dir[1], v_base_tn_dir[0]])
                chosen_avoid_dir = 0.6 * turn_dir + 0.6 * push_away_dir
                if np.linalg.norm(turn_dir) < 1e-6: chosen_avoid_dir = push_away_dir
                norm_chosen_avoid = np.linalg.norm(chosen_avoid_dir)
                if norm_chosen_avoid > 1e-6: v_avoid_vector = (chosen_avoid_dir / norm_chosen_avoid) * v_tn_max * strength_avoid
    return v_avoid_vector, is_avoiding

def _m5_sim2_combined_ode(t, state):
    z_kt = state[0:2]; z_tn = state[2:4]
    params = st.session_state.m5s2_params
    traj_params = st.session_state.m5s2_trajectory_params
    
    # --- Logic Tàu khu trục (pursuer) - GIỮ NGUYÊN ---
    dx_kt, dy_kt = 0.0, 0.0
    distance_kt_tn = np.linalg.norm(z_tn - z_kt)
    if 'm5s2_last_kt_dir' not in st.session_state: 
        st.session_state.m5s2_last_kt_dir = np.array([1.0, 0.0])
    if 0 < distance_kt_tn < params['kt_radar_radius']:
        if distance_kt_tn > params['catch_threshold'] / 2.0:
            dir_to_tn = (z_tn - z_kt) / distance_kt_tn
            dx_kt = params['v_kt'] * dir_to_tn[0]; dy_kt = params['v_kt'] * dir_to_tn[1]
            st.session_state.m5s2_last_kt_dir = dir_to_tn
    elif distance_kt_tn > params['catch_threshold'] / 2.0:
        dx_kt = (params['v_kt'] * 0.5) * st.session_state.m5s2_last_kt_dir[0]
        dy_kt = (params['v_kt'] * 0.5) * st.session_state.m5s2_last_kt_dir[1]

    # --- Logic Tàu ngầm (target) ---
    v_base_tn = _m5s2_get_base_submarine_velocity(t, traj_params, params['v_tn_max'])
    norm_v_base = np.linalg.norm(v_base_tn)
    v_base_dir = v_base_tn / norm_v_base if norm_v_base > 1e-6 else np.array([0.0, 0.0])
    
    v_avoid, is_avoiding = _m5s2_get_smarter_avoidance_info(z_tn, z_kt, v_base_dir, params['avoidance_radius'], params['v_tn_max'], params['avoidance_strength'], params['fov_tn_degrees'])
    
    if is_avoiding: 
        v_total_desired = 0.2 * v_base_tn + 0.8 * v_avoid
    else:
        if 'm5s2_last_free_turn' not in st.session_state: 
            st.session_state.m5s2_last_free_turn = t - params['min_time_free_turn'] * 2
        v_final = v_base_tn
        if (t - st.session_state.m5s2_last_free_turn) >= params['min_time_free_turn'] and norm_v_base > 1e-6:
            angle = random.uniform(-params['max_angle_free_turn_rad'], params['max_angle_free_turn_rad'])
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            v_final = np.dot(rot_matrix, v_base_tn)
            st.session_state.m5s2_last_free_turn = t
        v_total_desired = v_final
    
    # === SỬA LỖI LOGIC VẬN TỐC TÀU NGẦM (ĐỂ GIỐNG PYSIDE6) ===
    norm_total_tn = np.linalg.norm(v_total_desired)
    if norm_total_tn < 1e-9:
        dx_tn, dy_tn = 0.0, 0.0
    else:
        # Luôn chuẩn hóa và nhân với vận tốc tối đa
        dx_tn = (v_total_desired[0] / norm_total_tn) * params['v_tn_max']
        dy_tn = (v_total_desired[1] / norm_total_tn) * params['v_tn_max']
    
    return np.array([dx_kt, dy_kt, dx_tn, dy_tn])

@st.cache_data
def _run_and_cache_m5_sim2(_solver_func_name, t_array, initial_state, catch_radius, _params_dict_json, _traj_params_dict_json):
    solver_map = { 
        "AB2_system_M5_Sim2_CombinedLogic": AB2_system_M5_Sim2_CombinedLogic, "AB3_system_M5_Sim2_CombinedLogic": AB3_system_M5_Sim2_CombinedLogic, 
        "AB4_system_M5_Sim2_CombinedLogic": AB4_system_M5_Sim2_CombinedLogic, "AB5_system_M5_Sim2_CombinedLogic": AB5_system_M5_Sim2_CombinedLogic, 
        "AM2_system_M5_Sim2_CombinedLogic": AM2_system_M5_Sim2_CombinedLogic, "AM3_system_M5_Sim2_CombinedLogic": AM3_system_M5_Sim2_CombinedLogic, 
        "AM4_system_M5_Sim2_CombinedLogic": AM4_system_M5_Sim2_CombinedLogic 
    }
    method_func = solver_map[_solver_func_name]
    
    params = json.loads(_params_dict_json)
    traj_params = json.loads(_traj_params_dict_json)

    local_last_kt_dir = np.array([1.0, 0.0])
    local_last_free_turn = params['t_start'] - params['min_time_free_turn'] * 2

    def f_combined_for_solver_cached(t, current_state):
        nonlocal local_last_kt_dir, local_last_free_turn
        
        z_kt, z_tn = current_state[0:2], current_state[2:4]
        
        # Logic Tàu khu trục (pursuer) - GIỮ NGUYÊN
        dx_kt, dy_kt = 0.0, 0.0
        dist = np.linalg.norm(z_tn - z_kt)
        if 0 < dist < params['kt_radar_radius']:
            if dist > params['catch_threshold'] / 2.0:
                direction = (z_tn - z_kt) / dist
                dx_kt, dy_kt = params['v_kt'] * direction[0], params['v_kt'] * direction[1]
                local_last_kt_dir = direction
        elif dist > params['catch_threshold'] / 2.0:
            dx_kt = (params['v_kt'] * 0.5) * local_last_kt_dir[0]
            dy_kt = (params['v_kt'] * 0.5) * local_last_kt_dir[1]

        # Logic Tàu ngầm (target)
        v_base = _m5s2_get_base_submarine_velocity(t, traj_params, params['v_tn_max'])
        v_base_norm = np.linalg.norm(v_base)
        v_base_dir = v_base / v_base_norm if v_base_norm > 1e-6 else np.array([0.0, 0.0])
        
        v_avoid, is_avoiding = _m5s2_get_smarter_avoidance_info(z_tn, z_kt, v_base_dir, params['avoidance_radius'], params['v_tn_max'], params['avoidance_strength'], params['fov_tn_degrees'])
        
        if is_avoiding: 
            v_total_desired = 0.2 * v_base + 0.8 * v_avoid
        else:
            v_final = v_base
            if (t - local_last_free_turn) >= params['min_time_free_turn'] and v_base_norm > 1e-6:
                angle = random.uniform(-params['max_angle_free_turn_rad'], params['max_angle_free_turn_rad'])
                rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                v_final = np.dot(rot_matrix, v_base)
                local_last_free_turn = t
            v_total_desired = v_final
        
        # === SỬA LỖI LOGIC VẬN TỐC TÀU NGẦM (ĐỂ GIỐNG PYSIDE6) ===
        norm_total_tn = np.linalg.norm(v_total_desired)
        if norm_total_tn < 1e-9:
            dx_tn, dy_tn = 0.0, 0.0
        else:
            # Luôn chuẩn hóa và nhân với vận tốc tối đa
            dx_tn = (v_total_desired[0] / norm_total_tn) * params['v_tn_max']
            dy_tn = (v_total_desired[1] / norm_total_tn) * params['v_tn_max']
            
        return np.array([dx_kt, dy_kt, dx_tn, dy_tn])

    try:
        t_points, state_hist, caught, t_catch = method_func(
            f_combined_like=f_combined_for_solver_cached,
            t_array_full_potential=t_array,
            initial_state_combined=initial_state,
            catch_dist_threshold=catch_radius
        )
        return { "time_points": t_points, "state_history": state_hist, "caught": caught, "time_of_catch": t_catch }
    except Exception as e:
        print(f"Lỗi khi chạy mô phỏng M5 Sim 2 trong cache: {e}")
        return None

def run_and_store_model5_scenario2_results():
    st.session_state.m5s2_results = {} # Xóa kết quả cũ
    
    # Lấy dữ liệu từ validated_params, đây là điểm mấu chốt
    validated_params_from_s2 = st.session_state.get('validated_params', {})
    if not validated_params_from_s2:
        st.error("Lỗi nghiêm trọng: Không tìm thấy validated_params. Vui lòng quay lại trang mô phỏng và chạy lại.")
        return
        
    params_s2 = validated_params_from_s2.get('params', {})
    
    # ... (Toàn bộ phần còn lại của hàm được giữ nguyên y hệt)
    # --- 1. Lấy và thiết lập các tham số mô phỏng ---
    V_TN_MAX_SIM = params_s2.get('u', 3.0) 
    R_TN_PARAM_SIM = 2.0
    V_KT_SIM = params_s2.get('v', 6.0)
    INITIAL_DISTANCE_D_SIM = 30.0
    OMEGA_TN_PARAM_SIM = V_TN_MAX_SIM / R_TN_PARAM_SIM if R_TN_PARAM_SIM > 1e-6 else 0.1
    T_START_SIM = params_s2.get('t₀', 0.0)
    SOLVER_MAX_DURATION_GUESS = 70.0 
    T_END_SIM_S2 = T_START_SIM + SOLVER_MAX_DURATION_GUESS
    
    AVOIDANCE_RADIUS_SIM = INITIAL_DISTANCE_D_SIM * 0.40
    AVOIDANCE_STRENGTH_FACTOR_SIM = 1.1
    CATCH_THRESHOLD_SIM = 0.75
    FIELD_OF_VIEW_TN_DEGREES_SIM = 120.0
    KT_RADAR_RADIUS_SIM = AVOIDANCE_RADIUS_SIM * 2.8
    MIN_TIME_BETWEEN_FREE_TURNS_TN_SIM = SOLVER_MAX_DURATION_GUESS / 10.0
    FREE_TURN_ANGLE_MAX_RAD_TN_SIM = np.deg2rad(50)
    
    st.session_state.m5s2_params = {
        'v_kt': V_KT_SIM, 'z0_kt': np.array([params_s2.get('x0', 0.0), params_s2.get('y0', 0.0)]),
        'v_tn_max': V_TN_MAX_SIM,
        't_start': T_START_SIM, 't_end': T_END_SIM_S2, 
        'simulation_duration': SOLVER_MAX_DURATION_GUESS,
        'method_short': validated_params_from_s2.get('method_short', 'Bashforth'),
        'method_steps': validated_params_from_s2.get('selected_steps_int', [4])[-1],
        'avoidance_radius': AVOIDANCE_RADIUS_SIM,
        'avoidance_strength': AVOIDANCE_STRENGTH_FACTOR_SIM,
        'fov_tn_degrees': FIELD_OF_VIEW_TN_DEGREES_SIM,
        'kt_radar_radius': KT_RADAR_RADIUS_SIM,
        'min_time_free_turn': MIN_TIME_BETWEEN_FREE_TURNS_TN_SIM,
        'max_angle_free_turn_rad': FREE_TURN_ANGLE_MAX_RAD_TN_SIM,
        'catch_threshold': CATCH_THRESHOLD_SIM,
        'R_TN_PARAM': R_TN_PARAM_SIM, 'OMEGA_TN_PARAM': OMEGA_TN_PARAM_SIM
    }
    
    z0_kt_sim = st.session_state.m5s2_params['z0_kt']
    random_angle_init = random.uniform(0, 2 * np.pi)
    tn_offset_x = INITIAL_DISTANCE_D_SIM * np.cos(random_angle_init)
    tn_offset_y = INITIAL_DISTANCE_D_SIM * np.sin(random_angle_init)
    
    params_x_tn = [{"amp": random.uniform(12.0, 24.0), "freq": random.uniform(0.0015, 0.007), "phase": random.uniform(0, 2*np.pi), "type": 'sin'}]
    params_y_tn = [{"amp": random.uniform(10.0, 20.0), "freq": random.uniform(0.002, 0.008), "phase": random.uniform(0, 2*np.pi), "type": 'cos'}]
    
    sum_x_at_t0 = sum(p["amp"] * (np.sin(p["freq"] * T_START_SIM + p["phase"]) if p["type"] == 'sin' else np.cos(p["freq"] * T_START_SIM + p["phase"])) for p in params_x_tn)
    sum_y_at_t0 = sum(p["amp"] * (np.sin(p["freq"] * T_START_SIM + p["phase"]) if p["type"] == 'sin' else np.cos(p["freq"] * T_START_SIM + p["phase"])) for p in params_y_tn)

    offset_x_tn = z0_kt_sim[0] + tn_offset_x - sum_x_at_t0
    offset_y_tn = z0_kt_sim[1] + tn_offset_y - sum_y_at_t0

    st.session_state.m5s2_trajectory_params = {
        "offset_x": offset_x_tn, "offset_y": offset_y_tn,
        "params_x": params_x_tn, "params_y": params_y_tn
    }
    
    z_tn_actual_start = _m5s2_z_tn_base(T_START_SIM, st.session_state.m5s2_trajectory_params)

    params_json = json.dumps(st.session_state.m5s2_params, cls=NumpyEncoder)
    traj_params_json = json.dumps(st.session_state.m5s2_trajectory_params, cls=NumpyEncoder)

    method_short = st.session_state.m5s2_params['method_short']
    method_steps = st.session_state.m5s2_params['method_steps']
    
    solver_map_names = {
        "Bashforth": {2: "AB2_system_M5_Sim2_CombinedLogic", 3: "AB3_system_M5_Sim2_CombinedLogic", 4: "AB4_system_M5_Sim2_CombinedLogic", 5: "AB5_system_M5_Sim2_CombinedLogic"},
        "Moulton": {2: "AM2_system_M5_Sim2_CombinedLogic", 3: "AM3_system_M5_Sim2_CombinedLogic", 4: "AM4_system_M5_Sim2_CombinedLogic"}
    }
    solver_func_name = solver_map_names.get(method_short, {}).get(method_steps, "AB4_system_M5_Sim2_CombinedLogic")
    
    num_solver_steps = int(np.ceil(100 * st.session_state.m5s2_params['simulation_duration']))
    t_array_solver = np.linspace(st.session_state.m5s2_params['t_start'], st.session_state.m5s2_params['t_end'], num_solver_steps + 1)
    initial_state = np.array([z0_kt_sim[0], z0_kt_sim[1], z_tn_actual_start[0], z_tn_actual_start[1]])

    st.session_state.m5s2_results = _run_and_cache_m5_sim2(
        _solver_func_name=solver_func_name,
        t_array=t_array_solver,
        initial_state=initial_state,
        catch_radius=st.session_state.m5s2_params['catch_threshold'],
        _params_dict_json=params_json,
        _traj_params_dict_json=traj_params_json
    )
	
def create_animation_gif(lang_code, model_id, model_data, validated_params, speed_multiplier):
    # --- CÀI ĐẶT FONT VÀ HÀM DỊCH NGÔN NGỮ CỤC BỘ ---
    font_path = os.path.join(base_path, "fonts", "DejaVuSans.ttf")
    if os.path.exists(font_path):
        from matplotlib.font_manager import FontProperties
        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False 
    
    translations = load_language_file(lang_code)
    def _tr(key):
        return translations.get(key, key)

    progress_container = st.empty()
    with progress_container.container():
        progress_bar = st.progress(0)
        progress_text = st.empty()

    gif_buf = io.BytesIO()
    with imageio.get_writer(gif_buf, mode='I', format='gif', duration=0.1 / speed_multiplier, loop=None) as writer:
        try:
            fig, ax = plt.subplots(figsize=(10, 10), dpi=90)
            final_stats = {}

            # --- Lấy dữ liệu mô phỏng cần thiết ---
            sim_data = {}
            if model_id == 'model5' and st.session_state.m5_scenario == 1:
                # Chạy lại tính toán cho Sim 1 với solver và điều kiện dừng đúng
                validated_params_json = json.dumps(validated_params, cls=NumpyEncoder)
                sim_data = run_and_prepare_m5s1_animation_data(validated_params_json)
            elif model_id == 'model5' and st.session_state.m5_scenario == 2:
                if 'm5s2_results' not in st.session_state or not st.session_state.m5s2_results:
                    run_and_store_model5_scenario2_results()
                sim_data = st.session_state.get('m5s2_results', {})
            else: # Các model khác
                results = st.session_state.get('simulation_results', {})
                best_sim_data = None
                if results:
                    highest_step_found = -1
                    best_method_key = None
                    for method_key, step_results in results.items():
                        if step_results:
                            current_max_step = max(int(k) for k in step_results.keys())
                            if current_max_step > highest_step_found:
                                highest_step_found = current_max_step
                                best_method_key = method_key
                    if best_method_key is not None and highest_step_found != -1:
                        best_sim_data = results[best_method_key][highest_step_found]
                sim_data = best_sim_data if best_sim_data is not None else {}
            
            if not sim_data: return None, {}

            # --- Xác định số lượng frame tối đa ---
            num_frames = 0
            if model_id == 'model2': num_frames = len(sim_data.get('t_plot', []))
            elif model_id == 'model3': num_frames = model_data.get("abm_defaults", {}).get("max_steps", 400)
            elif model_id == 'model5' and st.session_state.m5_scenario == 1: num_frames = len(sim_data.get('t_plot', []))
            elif model_id == 'model5' and st.session_state.m5_scenario == 2: num_frames = len(sim_data.get('time_points', []))
            if num_frames == 0 and model_id != 'model3': return None, {}
            
            # --- Khởi tạo đối tượng mô phỏng ---
            abm_instance = None
            if model_id == 'model3':
                abm_params = model_data.get("abm_defaults", {})
                r_val = st.session_state.get('last_calculated_r', 0.0001)
                ptrans = np.clip(r_val * abm_params.get("r_to_ptrans_factor", 5000), abm_params.get("ptrans_min", 0.01), abm_params.get("ptrans_max", 0.9))
                total_pop = int(validated_params['params']['n'] + 1)
                abm_instance = DiseaseSimulationABM(
                    total_population=total_pop, initial_infected_count_for_abm=1,
                    room_dimension=abm_params.get('room_dimension', 10.0), 
                    contact_radius=abm_params.get('base_contact_radius', 0.5),
                    transmission_prob=ptrans, 
                    agent_speed=abm_params.get('base_agent_speed', 0.05)
                )
            model2_cells = [Cell(0, 0, gen=0)] if model_id == 'model2' else []
            
            # --- Vòng lặp tạo từng frame ---
            loop_iterator = range(num_frames)
            if model_id == 'model3':
                max_steps_abm = num_frames
                loop_iterator = range(max_steps_abm) # Chỉ cần range, logic break sẽ xử lý
            
            for frame_idx in loop_iterator:
                progress_percent = (frame_idx + 1) / num_frames if num_frames > 0 else 0
                progress_bar.progress(progress_percent)
                progress_text.text(f"{_tr('gif_generating_spinner')} ({frame_idx + 1}/{num_frames})")
                ax.clear()
                
                if model_id == 'model2':
                    t_data, y_data = sim_data['t_plot'], sim_data['approx_sol_plot']
                    target_n = int(round(y_data[frame_idx]))
                    while len(model2_cells) < target_n:
                        parent_cell = random.choice(model2_cells)
                        found_spot = False
                        for _ in range(20):
                            angle = random.uniform(0, 2 * np.pi); dist = 1.0
                            new_x, new_y = parent_cell.x + dist * np.cos(angle), parent_cell.y + dist * np.sin(angle)
                            if not any(np.hypot(new_x - c.x, new_y - c.y) < 1.0 for c in model2_cells):
                                model2_cells.append(Cell(new_x, new_y, parent_cell.gen + 1))
                                found_spot = True
                                break
                        if not found_spot: break
                    all_x = [c.x for c in model2_cells]; all_y = [c.y for c in model2_cells]
                    for cell in model2_cells: ax.add_patch(MplCircle((cell.x, cell.y), radius=0.5, color='#A52A2A', ec='black', lw=0.5, alpha=0.9))
                    if all_x:
                        max_coord = max(max(np.abs(all_x)), max(np.abs(all_y))) + 2
                        ax.set_xlim(-max_coord, max_coord); ax.set_ylim(-max_coord, max_coord)
                    ax.set_aspect('equal'); ax.axis('off')
                    ax.legend([MplCircle((0,0), 0.1, color='#A52A2A', ec='black', lw=0.5)], [_tr("screen3_legend_model2_cell")], loc='upper right')
                    ax.set_title(_tr("screen3_model2_anim_plot_title") + f"\nTime: {t_data[frame_idx]:.2f}s | Cells: {len(model2_cells)}")

                elif model_id == 'model3':
                    ended_by_logic = abm_instance.step()
                    ax.set_xlim(0, abm_instance.room_dimension); ax.set_ylim(0, abm_instance.room_dimension)
                    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
                    abm_params = model_data.get("abm_defaults", {})
                    s_coords, i_coords = abm_instance.get_display_coords(abm_params['display_max_total'], abm_params['display_sample_size'])
                    ax.set_facecolor('lightgray')
                    if s_coords.shape[0] > 0: ax.scatter(s_coords[:, 0], s_coords[:, 1], c='blue', s=30, label=_tr('screen3_legend_abm_susceptible'))
                    if i_coords.shape[0] > 0: ax.scatter(i_coords[:, 0], i_coords[:, 1], c='yellow', marker='*', s=70, edgecolors='red', label=_tr('screen3_legend_abm_infected'))
                    ax.legend()
                    stats = abm_instance.get_current_stats()
                    seconds_per_step = abm_params.get("seconds_per_step", 0.1) 
                    real_time_seconds = stats['time_step'] * seconds_per_step
                    title_text = _tr("screen3_model3_anim_plot_title"); time_label = _tr("screen3_actual_time"); infected_label = _tr("screen3_infected_label_short")
                    full_title = f"{title_text}\n{time_label}: {real_time_seconds:.2f}s | {infected_label}: {stats['infected_count']}"
                    ax.set_title(full_title, fontsize=9)
                    if ended_by_logic: break

                elif model_id == 'model5' and st.session_state.m5_scenario == 1:
                    t_data = sim_data.get('t_plot')
                    if sim_data.get('approx_sol_plot_all_components') is None: break
                    x_path, y_path = sim_data['approx_sol_plot_all_components']
                    d_val = validated_params['params']['x0']
                    
                    xlim, ylim = sim_data['plot_limits']['xlim'], sim_data['plot_limits']['ylim']
                    ax.set_xlim(xlim); ax.set_ylim(ylim)
                    
                    ax.fill_betweenx(ylim, xlim[0], 0, color='#A0522D', alpha=0.8)
                    ax.fill_betweenx(ylim, 0, d_val, color='#87CEEB', alpha=0.7)
                    ax.fill_betweenx(ylim, d_val, xlim[1], color='#A0522D', alpha=0.8)
                    
                    if sim_data.get('arrow_data'):
                        X, Y, U, V = sim_data['arrow_data']
                        ax.quiver(X, Y, U, V, color='blue', scale=25, width=0.004, headwidth=5, headlength=7, alpha=1.0)

                    line_ship_path, = ax.plot(x_path[:frame_idx+1], y_path[:frame_idx+1], '--', lw=2.5, color='darkslategray')
                    point_ship, = ax.plot(x_path[frame_idx], y_path[frame_idx], 
                                          marker='*', markersize=15, color='gold', 
                                          markeredgecolor='red', markeredgewidth=0.5)
                    ax.axhline(0, color='slategray', linestyle=':', linewidth=1.2, zorder=0.5)
                    ax.set_xlabel(_tr('screen3_model5_plot_xlabel_sim1')); ax.set_ylabel(_tr('screen3_model5_plot_ylabel_sim1'))
                    ax.grid(True, linestyle=':'); ax.set_aspect('equal')
                    ax.set_title(_tr("screen3_model5_plot_title_sim1") + f"\nTime: {t_data[frame_idx]:.2f}s")
                    
                    proxy_ship_legend = Line2D([0], [0], linestyle='None', marker='*', markersize=10, color='gold', markeredgecolor='red', markeredgewidth=0.5)
                    legend_handles = [line_ship_path, proxy_ship_legend]
                    legend_labels = [_tr('screen3_legend_m5s1_path'), _tr('screen3_legend_m5s1_boat')]
                    if sim_data.get('arrow_data'):
                         u_val = validated_params['params']['u']
                         arrow_marker = r'$\downarrow$' if u_val > 0 else (r'$\uparrow$' if u_val < 0 else '')
                         if arrow_marker:
                            proxy_arrow = Line2D([0], [0], linestyle='None', marker=arrow_marker, color='blue', markersize=10)
                            legend_handles.append(proxy_arrow)
                            legend_labels.append(_tr("screen3_legend_m5s1_water_current"))
                    ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right')

                elif model_id == 'model5' and st.session_state.m5_scenario == 2:
                    t_points, state_hist = sim_data['time_points'], sim_data['state_history']
                    is_caught, catch_time = sim_data['caught'], sim_data['time_of_catch']
                    pursuer_path, evader_path = state_hist[:, 0:2], state_hist[:, 2:4]
                    ax.plot(pursuer_path[:, 0], pursuer_path[:, 1], 'r-', label=_tr('screen3_legend_m5s2_path_destroyer'))
                    ax.plot(evader_path[:, 0], evader_path[:, 1], 'b--', label=_tr('screen3_legend_m5s2_path_submarine'))
                    ax.plot(pursuer_path[frame_idx, 0], pursuer_path[frame_idx, 1], 'rP', markersize=12, label=_tr('screen3_legend_m5s2_destroyer'))
                    ax.plot(evader_path[frame_idx, 0], evader_path[frame_idx, 1], 'bo', markersize=8, label=_tr('screen3_legend_m5s2_submarine'))
                    if is_caught and t_points[frame_idx] >= catch_time:
                        catch_frame_idx_arr = np.where(t_points >= catch_time)[0]
                        if len(catch_frame_idx_arr) > 0:
                            catch_frame_idx = catch_frame_idx_arr[0]
                            catch_point = state_hist[catch_frame_idx, 0:2]
                            ax.plot(catch_point[0], catch_point[1], 'gX', markersize=15, label=_tr('screen3_legend_m5s2_catch_point'))
                    ax.set_xlabel(_tr("screen3_model5_plot_xlabel_sim2")); ax.set_ylabel(_tr("screen3_model5_plot_ylabel_sim2"))
                    ax.grid(True); ax.legend(); ax.set_aspect('equal')
                    ax.set_title(_tr("screen3_model5_plot_title_sim2") + f"\nTime: {t_points[frame_idx]:.2f}s")
                    if is_caught and t_points[frame_idx] >= catch_time: break
                
                fig.canvas.draw()
                frame_buf = io.BytesIO()
                fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                fig.savefig(frame_buf, format='png')
                frame_buf.seek(0)
                writer.append_data(imageio.imread(frame_buf))
                frame_buf.close()

            plt.close(fig)

            # --- Tạo thông tin cuối cùng (final_stats) ---
            if model_id == 'model3':
                stats = abm_instance.get_current_stats()
                abm_defaults = model_data.get("abm_defaults", {})
                seconds_per_step = abm_defaults.get("seconds_per_step", 0.1)
                final_real_time = stats['time_step'] * seconds_per_step
                final_stats = {
                    _tr('screen3_total_pop'): {'value': stats['total_population']},
                    _tr('screen3_susceptible_pop'): {'value': stats['susceptible_count']},
                    _tr('screen3_infected_pop'): {'value': stats['infected_count']},
                    _tr('screen3_actual_time'): {'value': f"{final_real_time:.2f} s"} 
                }
            elif model_id == 'model2':
                c_val = st.session_state.get('last_calculated_c', 'N/A')
                final_stats = {
                    _tr('screen3_result_c'): {'value': f"{c_val:.4g}" if isinstance(c_val, float) else "N/A"},
                    _tr('screen3_result_mass'): {'value': len(model2_cells)},
                    _tr('screen3_result_time'): {'value': f"{sim_data['t_plot'][-1]:.2f} s"}
                }
            elif model_id == 'model5' and st.session_state.m5_scenario == 1:
                if sim_data.get('approx_sol_plot_all_components') is not None:
                    x_path, y_path = sim_data['approx_sol_plot_all_components']
                    final_stats = {
                        _tr('screen3_m5_boat_speed'): {'value': f"{validated_params['params']['v']:.2f}"},
                        _tr('screen3_m5_water_speed'): {'value': f"{validated_params['params']['u']:.2f}"},
                        _tr('screen3_m5_crossing_time'): {'value': f"{sim_data['t_plot'][-1]:.2f} s"},
                        _tr('screen3_m5_boat_reaches_target'): {'value': _tr('answer_yes') if abs(x_path[-1]) < 0.01 * validated_params['params']['x0'] else _tr('answer_no'), 'size_class': 'metric-value-small'},
                        _tr('screen3_m5_boat_final_pos'): {'value': f"({x_path[-1]:.2f}, {y_path[-1]:.2f})", 'size_class': 'metric-value-small'}
                    }
                else: final_stats = {}
            elif model_id == 'model5' and st.session_state.m5_scenario == 2:
                is_caught, catch_time = sim_data['caught'], sim_data['time_of_catch']
                status_str = _tr('answer_yes') if is_caught else _tr('answer_no')
                catch_point_str = "N/A"
                if is_caught:
                    catch_frame_idx_arr = np.where(sim_data['time_points'] >= catch_time)[0]
                    if len(catch_frame_idx_arr) > 0:
                        catch_frame_idx = catch_frame_idx_arr[0]
                        catch_point = sim_data['state_history'][catch_frame_idx, 0:2]
                        catch_point_str = f"({catch_point[0]:.2f}, {catch_point[1]:.2f})"
                final_stats = {
                    _tr('screen3_m5_submarine_speed'): {'value': f"{st.session_state.m5s2_params['v_tn_max']:.2f}"},
                    _tr('screen3_m5_destroyer_speed'): {'value': f"{st.session_state.m5s2_params['v_kt']:.2f}"},
                    _tr('screen3_m5_catch_time'): {'value': f"{sim_data['time_points'][-1]:.2f} s"},
                    _tr('screen3_m5_destroyer_catches_submarine'): {'value': status_str, 'size_class': 'metric-value-small'},
                    _tr('screen3_m5_catch_point'): {'value': catch_point_str, 'size_class': 'metric-value-small'}
                }
            else:
                final_stats = {}

        except Exception as e:
            print(f"Lỗi trong quá trình tạo frame GIF: {e}")
            import traceback
            traceback.print_exc()
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)
            return None, {}

    progress_container.empty()
    gif_buf.seek(0)
    return gif_buf.getvalue(), final_stats
def show_dynamic_simulation_page():
    # --- CSS và các hàm nội bộ ---
    st.markdown("""
    <style>
    .metric-container {
        border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem;
        margin-bottom: 1rem; background-color: #fafafa;
    }
    .metric-label { font-size: 1rem; color: #4a4a4a; margin-bottom: 0.5rem; font-weight: bold; }
    .metric-value { font-size: 1.5rem; color: #000000; font-weight: 600; line-height: 1.2; }
    .metric-value-small { font-size: 1.1rem; color: #000000; font-weight: 500; line-height: 1.2; }
    </style>
    """, unsafe_allow_html=True)

    def _cleanup_and_navigate(destination_page):
        """Dọn dẹp state của (các) trang liên quan và điều hướng đến trang mới."""
        # 1. Luôn dọn dẹp state của trang mô phỏng động (trang hiện tại)
        dynamic_keys_to_delete = [
            k for k in st.session_state 
            if k.startswith('anim_') or k.startswith('m5s') or k.startswith('gif_') 
            or k == 'abm_instance' or k == 'model2_cells' or k == 'generated_gif'
            or k == 'final_anim_stats' or k == 'generate_gif_request'
        ]
        for key in dynamic_keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        
        # 2. Luôn dọn dẹp state của trang mô phỏng tĩnh, vì nó là "trang mẹ"
        #    của trang mô phỏng động.
        st.session_state.simulation_results = {}
        st.session_state.validated_params = {}
        static_keys_to_clear = [k for k in st.session_state if k.startswith('last_calculated_')]
        for key in static_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # 3. Đặt trang mới và rerun để áp dụng thay đổi ngay lập tức
        st.session_state.page = destination_page
        st.rerun()
		
    def display_custom_metric(placeholder, data_dict):
        html_content = "<div class='metric-container'>"
        for label, value_info in data_dict.items():
            value, size_class = value_info['value'], value_info.get('size_class', 'metric-value')
            html_content += f"<div class='metric-label'>{label}</div><div class='{size_class}'>{value}</div>"
        html_content += "</div>"
        placeholder.markdown(html_content, unsafe_allow_html=True)

    # --- Kiểm tra dữ liệu đầu vào ---
    validated_params = st.session_state.get('validated_params_for_dynamic', {})
    if not validated_params or 'model_id' not in validated_params:
        st.error(tr("msg_no_data_for_dynamic"))
        if st.button(tr('screen3_back_button')): _cleanup_and_navigate('simulation')
        return

    model_id = validated_params.get("model_id")
    model_data = MODELS_DATA.get(st.session_state.get("selected_model_key"))
    is_processing = st.session_state.get('gif_is_processing', False)
    # --- Bố cục giao diện chính ---
    header_cols = st.columns([1.5, 4, 1.5])
    with header_cols[0]:
        back_cols = st.columns(2)
        if back_cols[0].button(f"ᐊ {tr('screen3_back_button')}", width='stretch', help=tr("screen3_dyn_back_tooltip"), disabled=is_processing):
            _cleanup_and_navigate('simulation')
        if back_cols[1].button(f"ᐊᐊ {tr('screen3_double_back_button')}", width='stretch', disabled=is_processing):
            _cleanup_and_navigate('model_selection')
    header_cols[1].markdown(f"<h1 style='text-align: center; margin: 0;'>{tr('screen3_dyn_only_title')}</h1>", unsafe_allow_html=True)
    
    col_controls, col_display = st.columns([1, 1.8])

    # --- Cột điều khiển ---
    with col_controls:
        with st.container(border=True):
            st.subheader(tr('screen3_settings_group_title'))
            
            speed_options = {
                tr("speed_slow"): 0.5,
                tr("speed_normal"): 1.0,
                tr("speed_fast"): 2.0,
                tr("speed_very_fast"): 4.0
            }
            selected_speed_label = st.selectbox(
                tr("screen3_speed_label"),
                options=speed_options.keys(),
                index=1,
                key="gif_speed_selector",
				disabled=is_processing
            )
            speed_multiplier = speed_options[selected_speed_label]
            st.session_state.speed_multiplier = speed_multiplier

            # Highlight: Sửa logic của nút bấm
            if st.button(f"🚀 {tr('generate_and_show_button')}", width='stretch', type="primary", disabled=is_processing,key="regenerate_gif_btn"):
                # Chỉ đặt cờ, không rerun
                st.session_state.generate_gif_request = True
                if 'generated_gif' in st.session_state:
                    del st.session_state['generated_gif']
                st.rerun()
        
        if model_id == 'model5':
            with st.container(border=True):
                if 'm5_scenario' not in st.session_state: st.session_state.m5_scenario = 1
                scenario_options = {tr("screen3_sim1_name_m5"): 1, tr("screen3_sim2_name_m5"): 2}
                def on_scenario_change():
                    # Hàm này sẽ được gọi KHI người dùng chọn radio button mới
                    keys_to_delete = [k for k in st.session_state if k.startswith('m5s') or k == 'generated_gif' or k == 'final_anim_stats']
                    for k in keys_to_delete:
                        if k in st.session_state:
                            del st.session_state[k]
                selected_scenario_disp = st.radio(
                    tr("screen3_sim_list_group_title"), 
                    options=scenario_options.keys(), 
                    index=st.session_state.m5_scenario - 1, 
                    key="m5_scenario_selector",
                    on_change=on_scenario_change,
					disabled=is_processing
                )
                st.session_state.m5_scenario = scenario_options[selected_scenario_disp]

        with st.container(border=True):
            info_title_key = "screen3_results_group_title"
            current_scenario = st.session_state.get('m5_scenario', 1)
            if model_id == 'model5':
                info_title_key = "screen3_info_m5_sim1_title" if current_scenario == 1 else "screen3_info_m5_sim2_title"
            
            st.subheader(tr(info_title_key))
            info_placeholder = st.empty()
            with info_placeholder.container():
                st.info(tr("press_generate_to_see_info"))

    # --- Cột hiển thị chính ---
    with col_display:
        # Highlight: Sửa lại logic hiển thị
        # Ưu tiên kiểm tra cờ yêu cầu tạo GIF trước
        if st.session_state.get('generate_gif_request', False):
            speed_multiplier = st.session_state.get('speed_multiplier', 1.0)
            # Hàm create_animation_gif sẽ tự điền vào các placeholder nó tạo ra
            gif_bytes, final_stats = gif_bytes, final_stats = create_animation_gif(
                st.session_state.lang, # Truyền mã ngôn ngữ hiện tại
                model_id, 
                model_data, 
                validated_params, 
                speed_multiplier
            )
            
            st.session_state.generate_gif_request = False # Reset cờ
            st.session_state.gif_is_processing = False
            if gif_bytes:
                st.session_state.generated_gif = gif_bytes
                st.session_state.final_anim_stats = final_stats
                st.rerun()
            else:
                st.error(tr("gif_generation_error"))
                info_placeholder.error(tr("gif_generation_error"))
                st.rerun()

        elif 'generated_gif' in st.session_state and st.session_state.generated_gif:
            # Nếu đã có GIF, hiển thị nó
            st.image(st.session_state.generated_gif)
            final_stats = st.session_state.get('final_anim_stats', {})
            if final_stats:
                display_custom_metric(info_placeholder, final_stats)
        else:
            # Trạng thái ban đầu
            plot_placeholder = st.empty()
            with plot_placeholder.container():
                fig, ax = plt.subplots(figsize=(8,8))
                ax.text(0.5, 0.5, tr("press_generate_to_see_info"), ha='center', va='center')
                ax.set_xticks([]); ax.set_yticks([])
                st.pyplot(fig)
            with info_placeholder.container():
                st.info(tr("press_generate_to_see_info"))

# =========================================================================
# Highlight: KẾT THÚC CẬP NHẬT VÒNG LẶP ANIMATION
# =========================================================================


# Điểm bắt đầu chạy ứng dụng
if __name__ == "__main__":
    main()
