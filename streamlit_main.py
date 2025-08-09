# ==============================================
#           PHẦN 1: CÀI ĐẶT VÀ CÁC HÀM LÕI
# ==============================================

# --- 1. IMPORTS ---
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as MplCircle 
import os
from datetime import datetime
import random
import time # Cần cho animation

# Giả định file ảnh và các tài nguyên khác nằm cùng thư mục với script
base_path = os.path.dirname(os.path.abspath(__file__))
# Highlight: Đường dẫn đến thư mục chứa ảnh
FIG_FOLDER = os.path.join(base_path, "fig")

# --- 2. CÁC BIẾN CỐ ĐỊNH VÀ TỪ ĐIỂN NGÔN NGỮ (GIỮ NGUYÊN) ---

# Giả định file ảnh và các tài nguyên khác nằm cùng thư mục với script
base_path = os.path.dirname(os.path.abspath(__file__))

LANG_VI = {
	"app_title": "Ứng dụng mô phỏng phương pháp đa bước - FMS TDTU", 
	# Welcome Screen
	"welcome_uni": "TRƯỜNG ĐẠI HỌC TÔN ĐỨC THẮNG",
	"welcome_faculty": "KHOA TOÁN - THỐNG KÊ", 
	"welcome_project_title": "Phương pháp đa bước và ứng dụng\ntrong mô phỏng một số mô hình thực tế", 
	"welcome_authors_title": "Tác giả",
	"welcome_authors_names": "   Khưu Tấn Đạt – Huỳnh Nhựt Trường – Đào Nhật Gia Ân", 
	"welcome_advisors_title": "Giảng viên hướng dẫn",
	"welcome_advisor1": "PGS. TS. Trần Minh Phương", 
	"welcome_advisor2": "TS. Nguyễn Hữu Cần", 
	"lang_vi": "Tiếng Việt",
	"lang_en": "English", 
	"start_button": "Bắt đầu",
	# Screen 1
	"screen1_title": "Chọn mô hình thực tế",
	"screen1_model_info_group_title": "Thông tin mô hình",
	"screen1_model_application_group_title": "Ứng dụng của mô hình",
	"screen1_equation_label": "Phương trình vi phân/Nghiệm giải tích:",
	"screen1_description_label": "Mô tả và ý nghĩa tham số:",
	"screen1_continue_button": "Tiếp tục với mô hình này",
	# ================== MODEL 1: Energy Demand ==================
	"model1_name": "Mô hình 1: Nguồn năng lượng",
	"model1_eq": "dO/dt = kO<br>O(t) = O<sub>0</sub>e<sup>kt</sup>",#-t<sub>0</sub>
	"model1_desc": ("Mô hình tăng trưởng theo hàm mũ, mô tả mức tiêu thụ năng lượng theo thời gian.<br>"
	"O(t): Mức tiêu thụ năng lượng tại thời điểm t.<br>"
	"dO/dt: Tốc độ tăng trưởng mức tiêu thụ năng lượng.<br>"
	"O<sub>0</sub>: Mức tiêu thụ năng lượng ban đầu tại thời điểm t<sub>0</sub>.<br>"
	"k: Hằng số biểu thị tốc độ tăng trưởng.<br>"),
	#"t<sub>0</sub>: Thời điểm ban đầu."),
	"model1_param1": "Mức tiêu thụ năng lượng ban đầu (O₀)",
	"model1_param2": "Hằng số tốc độ tăng trưởng (k)",
	"model1_param3": "Thời điểm ban đầu (t₀)",
	"model1_param4": "Thời điểm kết thúc (t₁)",
	# ================== MODEl 2 ==================
	"model2_name": "Mô hình 2: Sự tăng trưởng tế bào",
	"model2_eq": "dx/dt = cx<sup>2/3</sup><br>x(t) = (x<sub>0</sub><sup>1/3</sup> + ct/3)<sup>3</sup>",
	"model2_desc": ("Mô hình mô phỏng sự tăng trưởng khối lượng của tế bào.<br>"
	"x(t): Khối lượng của tế bào tại thời điểm t.<br>"
	"c: Hằng số biểu thị tốc độ tăng trưởng của tế bào.<br>"
	"dx/dt: Tốc độ thay đổi của khối lượng tế bào theo thời gian.<br>"
	"x<sub>0</sub>: Khối lượng ban đầu của tế bào (tại t=0 nếu nghiệm là ct/3)."),
	"model2_param1": "Khối lượng ban đầu (x₀)",
	"model2_param3": "Thời điểm ban đầu (t₀)",
	"model2_param4": "Thời điểm kết thúc (t₁)",
	"model2_calculated_c_label": "Hằng số tốc độ tăng trưởng (c):",
	# ================== MODEL 3: Spread of epidemic ==================
	"model3_name": "Mô hình 3: Sự lây lan dịch bệnh",
	"model3_eq": "y'(t) = -r*y(t)*(n+1-y(t))<br>y(t)  = n(n+1)e<sup>-r(n+1)(t-t<sub>0</sub>)</sup>/(1+ne<sup>-r(n+1)(t-t<sub>0</sub>)</sup>)",
	"model3_desc": ("Mô hình mô phỏng sự thay đổi về số lượng cá thể chưa bị nhiễm bệnh trong một cộng đồng theo thời gian.<br>" 
	"x(t): Số lượng cá thể chưa bị nhiễm tại thời điểm t.<br>"
	"n: Số lượng cá thể dễ bị nhiễm ban đầu.<br>" 
	"r: Hằng số dương đo lường tốc độ lây nhiễm.<br>"
	"dx/dt: Tốc độ giảm của số người chưa nhiễm (tốc độ lây nhiễm).<br>" 
	"t<sub>0</sub>: Thời điểm ban đầu."),
	"model3_param2": "Số lượng cá thể dễ bị nhiễm ban đầu (n)",
	"model3_param4": "Thời điểm ban đầu (t₀)",
	"model3_param5": "Thời điểm kết thúc (t₁)",
	"model3_calculated_r_label": "Hằng số tốc độ lây nhiễm (r):",
	# ================== MODEL 4: National Economy ==================
	"model4_name": "Mô hình 4: Nền kinh tế quốc gia",
	"model4_eq": "Y''(t) + αY'(t) + βY(t) = mlG", 
	"model4_desc": ("Mô hình mô phỏng sự thay đổi của nền kinh tế theo thời gian dựa trên các biến số của kinh tế vĩ mô.<br>"
	"Y(t): Thu nhập quốc dân.<br>"
	"Y'(t): Tốc độ thay đổi thu nhập.<br>"
	"α: Hệ số liên quan đến khuynh hướng tiêu dùng/đầu tư (α = m + ls - lma).<br>"
	"β: Hệ số phản ứng đầu tư theo thay đổi thu nhập (β = lms).<br>" 
	"m: Số nhân chi tiêu chính phủ.<br>"
	"l: Tham số liên quan đến cầu tiền.<br>"
	"a: Tham số độ nhạy đầu tư theo lãi suất.<br>"
	"s: Tham số độ nhạy cầu tiền theo lãi suất.<br>"
	"G: Chi tiêu chính phủ (hằng số)."),
	"model4_param_alpha": "Hệ số α", 
	"model4_param_beta": "Hệ số β", 
	"model4_param_a": "Tham số a",
	"model4_param_s": "Tham số s",
	"model4_param_m": "Số nhân m",
	"model4_param_G": "Chi tiêu G",
	"model4_param_l": "Tham số l",
	"model4_param_Y0": "Y(t₀)",
	"model4_param_dY0": "Y'(t₀)",
	"model4_param_t0": "Thời gian t₀",
	"model4_param_t1": "Thời gian t₁",
	# ================== MODEL 5: Pursuit Curve ==================
	"model5_name": "Mô hình 5: Đường cong truy đuổi",
	"model5_eq": "dx/dt = -v * x / √(x<sup>2</sup>+y<sup>2</sup>)<br>dy/dt = -v * y / √(x<sup>2</sup>+y<sup>2</sup>) - u",
	"model5_desc": ("Mô hình tìm quỹ đạo của con thuyền di chuyển dưới tác động của dòng chảy.<br>"
	"(x(t), y(t)): Tọa độ của con thuyền tại thời điểm t.<br>"
	"v: Vận tốc của con thuyền (tương đối so với nước).<br>"
	"u: Vận tốc của dòng chảy (theo chiều âm trục y).<br>"
	"x<sub>0</sub>, y<sub>0</sub>: Tọa độ ban đầu của con thuyền.<br>"
	"t<sub>0</sub>, t<sub>1</sub>: Khoảng thời gian mô phỏng."),
	"model5_param_x0": "Tọa độ x ban đầu (x₀)",
	"model5_param_y0": "Tọa độ y ban đầu (y₀)",
	"model5_param_u": "Vận tốc dòng chảy (u)",
	"model5_param_v": "Vận tốc thuyền (v)",
	"model5_param_t0": "Thời điểm bắt đầu (t₀)",
	"model5_param_t1": "Thời điểm kết thúc (t₁)",
	"model5_select_component": "Chọn thành phần để hiển thị:",
	"model5_component_x": "Thành phần X",
	"model5_component_y": "Thành phần Y",
	# Screen 2
	"screen2_back_button": "Quay lại chọn mô hình", 
	"screen2_goto_screen3_button": "Xem mô phỏng động", 
	"screen2_goto_screen3_tooltip": "Chuyển đến màn hình mô phỏng động (nếu có)",
	"screen2_method_group": "1. Chọn phương pháp chính",
	"screen2_method_ab": "Adams-Bashforth (AB)",
	"screen2_method_am": "Adams-Moulton (AM)",
	"screen2_details_group_ab": "2. Chi tiết Adams-Bashforth",
	"screen2_details_group_am": "2. Chi tiết Adams-Moulton",
	"screen2_steps_label": "Số bước:",
	"screen2_select_all_steps_cb": "Tất cả",
	"screen2_step2": "2 bước",
	"screen2_step3": "3 bước",
	"screen2_step4": "4 bước",
	"screen2_step5": "5 bước",
	"param_placeholder": "Nhập giá trị số...",
	"screen2_h_label": "Bước nhảy (h):",
	"screen2_sim_toggle": "Hiển thị cửa sổ mô phỏng động riêng",
	"screen2_params_group": "3. Tham số đầu vào của mô hình",
	"screen2_actions_group": "4. Thực thi và lưu trữ", 
	"screen2_init_button": "🚀 Khởi tạo và vẽ đồ thị", 
	"screen2_refresh_button": "🔄 Làm mới thông số",
	"screen2_show_data_button": "📊 Xem dữ liệu số",
	"screen2_save_button": "💾 Lưu hình ảnh đồ thị",
	"screen2_plot_solution_title": "Đồ thị nghiệm",
	"screen2_plot_error_title": "Đồ thị sai số L∞",
	"screen2_plot_order_title": "Đồ thị bậc hội tụ",
	"screen2_plot_t_axis": "Thời gian (t)",
	"screen2_plot_value_axis": "Giá trị nghiệm",
	"screen2_plot_n_axis": "Số khoảng con (N)",
	"screen2_plot_h_axis": "Bước nhảy (h)",
	"screen2_plot_error_axis": "Sai số L∞",
	"screen2_plot_log_h_axis": "log(h)",
	"screen2_plot_log_error_axis": "log(Sai số L∞)",
	"screen2_plot_exact_label": "Nghiệm chính xác / Tham chiếu",
	"screen2_plot_error_label_prefix": "Sai số ",
	"screen2_plot_order_data_label_suffix": " (dữ liệu)",
	"screen2_plot_order_fit_label_suffix": " Fit: O(h<sup>{:.2f}</sup>)",
	"screen2_info_area_init": "Nhấn 'Khởi tạo và vẽ đồ thị' để bắt đầu mô phỏng.", 
	"screen2_info_area_running": "Đang xử lý, vui lòng chờ...",
	"screen2_info_area_error": "Lỗi: ",
	"screen2_info_area_no_results": "Không có kết quả mô phỏng để hiển thị.",
	"screen2_info_area_no_show_data": "Chưa có dữ liệu. Vui lòng chạy mô phỏng trước.",
	"screen2_info_area_show_data_method": "Phương pháp:",
	"screen2_info_area_show_data_textCont1": "bước",
	"screen2_info_area_show_data_textCont2": "với",
	"screen2_info_area_show_data_order": "Bậc hội tụ ước tính:",
	"screen2_info_area_show_data_points_header": "Bảng sai số",
	"screen2_info_area_show_data_time": "t",
	"screen2_info_area_show_data_approx": "Xấp xỉ", 
	"screen2_info_area_show_data_exact": "Chính xác", 
	"screen2_info_area_show_data_error": "Sai số",
	"screen2_info_area_show_data_more": "(và các điểm khác...)",
	"screen2_info_area_show_data_no_points": "(Không có dữ liệu điểm)",
	"screen2_info_area_refreshed": "Đã làm mới. Sẵn sàng cho mô phỏng mới.",
	"screen2_info_area_complete": "Hoàn thành!",
	"screen2_legend_title": "Chú thích",
	"screen2_legend_order_title": "Bậc hội tụ:",
	"screen2_plot_value_axis_base": "Giá trị",
	"screen2_plot_value_axis_x": "Giá trị (thành phần X)", 
	"screen2_plot_value_axis_y": "Giá trị (thành phần Y)", 
	"screen2_plot_error_axis_base": "Sai số L∞",
	"screen2_plot_error_axis_x": "Sai số L∞ (thành phần X)", 
	"screen2_plot_error_axis_y": "Sai số L∞ (thành phần Y)", 
	"screen2_plot_log_error_axis_base": "log(Sai số L∞)",
	"screen2_plot_log_error_axis_x": "log(Sai số L∞ X)",
	"screen2_plot_log_error_axis_y": "log(Sai số L∞ Y)",
	"screen2_info_area_show_data_approx_x": "Xấp xỉ",    
	"screen2_info_area_show_data_approx_y": "Xấp xỉ",    
	"screen2_info_area_show_data_exact_x": "Tham chiếu", 
	"screen2_info_area_show_data_exact_y": "Tham chiếu", 
	"screen2_info_area_show_data_error_x": "Sai số",     
	"screen2_info_area_show_data_error_y": "Sai số",     	
	# Simulation Plot Window
	"sim_window_title": "Cửa sổ đồ thị mô phỏng động", 
	"sim_window_plot_title": "Mô phỏng nghiệm theo thời gian", 
	"sim_window_t_axis": "Thời gian (t)",
	"sim_window_value_axis": "Giá trị nghiệm", 
	"sim_window_no_data": "Không có dữ liệu hợp lệ để vẽ đồ thị động",
	# Messages
	"msg_error": "Lỗi",
	"msg_warning": "Cảnh báo",
	"msg_info": "Thông báo",
	"msg_select_method": "Vui lòng chọn một phương pháp.",
	"msg_select_step": "Vui lòng chọn ít nhất một số bước.",
	"msg_select_h": "Vui lòng chọn một giá trị bước nhảy (h).",
	"msg_invalid_h": "Giá trị bước nhảy h không hợp lệ: '{}'.",
	"msg_invalid_input_title": "Lỗi nhập liệu",
	"msg_invalid_input_text": "Vui lòng kiểm tra lại các tham số đầu vào.",
	"msg_invalid_param_value": "Tham số '{}' không phải là một số hợp lệ (giá trị nhận được: '{}').",
	"msg_missing_param_value": "Tham số '{}' không được để trống.",
	"msg_model_error_title": "Lỗi mô hình", 
	"msg_model_no_ode": "Mô hình '{}' chưa được định nghĩa hàm PTVP.",
	"msg_param_error_title": "Lỗi tham số", 
	"msg_missing_keys": "Thiếu các tham số bắt buộc: {}",
	"msg_missing_y0": "Thiếu điều kiện ban đầu (ví dụ: O₀, x₀, Y(t₀)).",
	"msg_param_value_error_title": "Lỗi giá trị tham số", 
	"msg_param_value_error_text": "Lỗi giá trị trong tham số: {}",
	"msg_t_end_error": "Thời điểm kết thúc t₁ phải lớn hơn thời điểm bắt đầu t₀.",
	"msg_unknown_error_title": "Lỗi không xác định",
	"msg_unknown_error_prep": "Đã xảy ra lỗi khi chuẩn bị cho mô phỏng: {}",
	"msg_internal_error_title": "Lỗi nội bộ",
	"msg_internal_error_steps": "Giá trị số bước không hợp lệ: {}",
	"msg_simulation_error_title": "Lỗi trong quá trình mô phỏng",
	"msg_simulation_error_text": "Đã xảy ra lỗi trong quá trình chạy mô phỏng: {}",
	"msg_no_results_title": "Không có kết quả",
	"msg_no_results_text": "Không tạo được kết quả mô phỏng hợp lệ.",
	"msg_sim_window_no_data": "Không có dữ liệu để hiển thị trong cửa sổ mô phỏng động.",
	"msg_show_data_no_data": "Chưa có dữ liệu mô phỏng để hiển thị.",
	"msg_save_no_plots": "Chưa có đồ thị nào được tạo để lưu.",
	"msg_save_select_dir": "Vui lòng chọn thư mục để lưu hình ảnh",
	"msg_save_cancelled": "Đã hủy thao tác chọn thư mục.",
	"msg_save_success_title": "Lưu thành công",
	"msg_save_success_text": "Đã lưu {} hình ảnh vào thư mục:\n{}",
	"msg_save_error_title": "Lỗi khi lưu",
	"msg_save_error_text": "Đã lưu {} hình ảnh nhưng gặp một số lỗi:\n{}",
	"msg_save_fail_title": "Lưu thất bại",
	"msg_save_fail_text": "Không thể lưu hình ảnh do lỗi:\n{}",
	"msg_saving_plot_error": "Lỗi khi lưu đồ thị '{}': {}",
	"msg_skipping_save": "Bỏ qua việc lưu đồ thị '{}' (không có dữ liệu).",
	
	#Screen 3
	"screen3_back_button": "Quay lại nhập tham số", 
	"screen3_double_back_button": "Quay lại chọn mô hình", 
	"screen3_dyn_only_title": "Mô phỏng động của mô hình",
	"screen3_dyn_back_tooltip": "Quay lại màn hình nhập liệu và các đồ thị kết quả tĩnh",
	"screen3_settings_group_title": "Cài đặt mô phỏng động",
	"screen3_speed_label": "Tốc độ phát lại:",
	"screen3_results_group_title": "Thông tin mô phỏng động",
	"screen3_unified_results_title": "Thông tin mô phỏng",
	"screen3_stop_button": "⏹ Dừng/Tiếp tục mô phỏng", 
	"screen3_result_time": "Thời gian mô phỏng (t):",
	"screen3_result_c": "Hằng số tăng trưởng (c):",
	"screen3_result_mass": "Số lượng tế bào (hoặc khối lượng):",
	"screen3_legend_model2_cell": "Tế bào",
	"screen3_waiting_for_data": "Đang chờ dữ liệu hoặc bắt đầu mô phỏng...",
	"screen3_result_r_param": "Tham số tốc độ lây nhiễm (r):",
	"screen3_model3_simulation_time_label": "Thời gian mô phỏng ABM (bước):",
	"screen3_actual_time": "Thời gian thực tế (giây):",
	"screen3_total_pop": "Tổng số cá thể ban đầu:",
	"screen3_infected_pop": "Số lượng cá thể bị nhiễm:",
	"screen3_susceptible_pop": "Số lượng cá thể chưa nhiễm:",
	"screen3_legend_abm_susceptible": "Chưa nhiễm",
	"screen3_legend_abm_infected": "Bị nhiễm",
	
	"screen3_sim_list_group_title": "Danh sách mô phỏng",
	"screen3_sim1_name_m5": "Mô phỏng 1: Bài toán con thuyền sang sông nhằm đích cố định",
	"screen3_sim2_name_m5": "Mô phỏng 2: Bài toán tàu khu trục lùng bắt tàu ngầm (trong sương mù)",
	
	"screen3_info_m5_sim1_title": "Thông tin mô phỏng 1 (con thuyền)", 
	"screen3_m5_boat_speed": "Vận tốc thuyền (v):",
	"screen3_m5_water_speed": "Vận tốc dòng nước (u):",
	"screen3_m5_crossing_time": "Thời gian mô phỏng (t):",
	"screen3_m5_start_point_boat": "Điểm bắt đầu của thuyền:",
	"screen3_m5_boat_reaches_target": "Thuyền có đến được đích không?",
	"screen3_m5_boat_final_pos": "Vị trí cuối cùng của thuyền:",
	"screen3_m5_determining_status": "Đang xác định...",
	"answer_yes": "Có",
	"answer_no": "Không",
	
	"screen3_info_m5_sim2_title": "Thông tin mô phỏng 2 (tàu khu trục vs tàu ngầm)", 
	"screen3_m5_submarine_speed": "Vận tốc tàu ngầm (v<sub>target</sub>):",
	"screen3_m5_destroyer_speed": "Vận tốc tàu khu trục (v<sub>pursuer</sub>):",
	"screen3_legend_m5s2_tn_avoid_radius": "Radar của tàu ngầm",
	"screen3_legend_m5s2_kt_radar_radius": "Radar của tàu khu trục",
	"screen3_m5_submarine_trajectory": "Phương trình quỹ đạo cơ sở của tàu ngầm:",
	"screen3_m5_start_point_submarine": "Điểm bắt đầu của tàu ngầm:",
	"screen3_m5_start_point_destroyer": "Điểm bắt đầu của tàu khu trục:",
	"screen3_m5_destroyer_catches_submarine": "Tàu khu trục có bắt được tàu ngầm không?",
	"screen3_m5_catch_point": "Điểm bắt được (tọa độ):",
	"screen3_m5_catch_time": "Thời gian mô phỏng:", 
	"screen3_model5_not_implemented_msg": "Mô phỏng cho kịch bản này chưa được triển khai.",
	
	"screen3_model2_anim_plot_title": "Mô phỏng động: Sự tăng trưởng tế bào", 
	"screen3_abm_anim_plot_title": "Mô phỏng động: Sự lây lan dịch bệnh (ABM)", 
	
	"screen3_model5_plot_title_sim1": "Mô phỏng động: Con thuyền qua sông", 
	"screen3_model5_plot_title_sim2": "Mô phỏng động: Tàu khu trục - Tàu ngầm", 
	
	"screen3_model5_plot_xlabel_sim1": "Tọa độ X (m)",
	"screen3_model5_plot_ylabel_sim1": "Tọa độ Y (m)",
	"screen3_legend_m5s1_path": "Quỹ đạo thuyền",
	"screen3_legend_m5s1_boat": "Con thuyền",
	"screen3_legend_m5s1_water_current": "Hướng dòng nước",
	
	"screen3_model5_plot_xlabel_sim2": "Tọa độ X (m)",
	"screen3_model5_plot_ylabel_sim2": "Tọa độ Y (m)",
	"screen3_legend_m5s2_submarine": "Tàu ngầm",
	"screen3_legend_m5s2_destroyer": "Tàu khu trục",
	"screen3_legend_m5s2_path_submarine": "Quỹ đạo tàu ngầm",
	"screen3_legend_m5s2_path_destroyer": "Quỹ đạo tàu khu trục",
	"screen3_legend_m5s2_catch_point": "Điểm bắt được",
	"screen3_m5_time_when_caught_label": "Thời gian bắt được tàu ngầm:",
}

# ========================================================================================================

LANG_EN = {
	"app_title": "Multi-step methods simulation application - FMS TDTU", 
	# Welcome Screen
	"welcome_uni": "TON DUC THANG UNIVERSITY",
	"welcome_faculty": "FACULTY OF MATHEMATICS - STATISTICS",
	"welcome_project_title": "Multi-step methods and applications\n in simulating some real-life models", 
	"welcome_authors_title": "Authors",
	"welcome_authors_names": "Tan Dat Khuu – Nhut Truong Huynh – Nhat Gia An Dao",
	"welcome_advisors_title": "Advisors",
	"welcome_advisor1": "Assoc. Prof. Minh Phuong Tran",
	"welcome_advisor2": "PhD. Huu Can Nguyen",
	"lang_vi": "Tiếng Việt", 
	"lang_en": "English",
	"start_button": "Start",
	# Screen 1
	"screen1_title": "Select a real-life model", 
	"screen1_model_info_group_title": "Model information", 
	"screen1_model_application_group_title": "Model application", 
	"screen1_equation_label": "Differential equation / Analytical solution:", 
	"screen1_description_label": "Description and parameters:", 
	"screen1_continue_button": "Continue with this model", 
	# ================== MODEL 1: Energy Demand ==================
	"model1_name": "Model 1: Energy demand", 
	"model1_eq": "dO/dt = kO<br>O(t) = O<sub>0</sub>e<sup>kt</sup>",#-t<sub>0</sub>
	"model1_desc": ("Exponential growth model describing energy consumption over time.<br>"
	"O(t): Energy consumption level at time t.<br>"
	"dO/dt: Rate of growth of energy consumption.<br>"
	"O<sub>0</sub>: Initial energy consumption level at time t<sub>0</sub>.<br>"
	"k: Constant representing the growth rate.<br>"),
	#"t<sub>0</sub>: Initial time."),
	"model1_param1": "Initial energy consumption (O₀)",
	"model1_param2": "Growth rate constant (k)",
	"model1_param3": "Initial time (t₀)",
	"model1_param4": "End time (t₁)",
	# ================== MODEl 2 ==================
	"model2_name": "Model 2: Cell growth", 
	"model2_eq": "dx/dt = cx<sup>2/3</sup><br>x(t) = (x<sub>0</sub><sup>1/3</sup> + ct/3)<sup>3</sup>",
	"model2_desc": ("Model simulating the growth of cell mass.<br>"
	"x(t): Mass of the cell at time t.<br>"
	"c: Constant representing the cell growth rate.<br>"
	"dx/dt: Rate of change of cell mass over time.<br>"
	"x<sub>0</sub>: Initial mass of the cell (at t=0 if solution is ct/3)."),
	"model2_param1": "Initial mass (x₀)",
	"model2_param3": "Initial time (t₀)",
	"model2_param4": "End time (t₁)",
	"model2_calculated_c_label": "Growth rate constant (c):",
	# ================== MODEL 3: Spread of epidemic ==================
	"model3_name": "Model 3: Spread of epidemic", 
	"model3_eq": "y'(t) = -r*y(t)*(n+1-y(t))<br>y(t)  = n(n+1)e<sup>-r(n+1)(t-t<sub>0</sub>)</sup>/(1+ne<sup>-r(n+1)(t-t<sub>0</sub>)</sup>)",
	"model3_desc": ("Model simulating the change in the number of uninfected individuals in a community over time.<br>"
	"x(t): Number of uninfected individuals at time t.<br>"
	"n: Initial number of susceptible individuals.<br>"
	"r: Positive constant measuring the rate of infection.<br>"
	"dx/dt: Rate of decrease of uninfected individuals (infection rate).<br>"
	"t<sub>0</sub>: Initial time."),
	"model3_param2": "Initial number of susceptible individuals (n)",
	"model3_param4": "Initial time (t₀)",
	"model3_param5": "End time (t₁)",
	"model3_calculated_r_label": "Infection rate constant (r):",
	# ================== MODEL 4: National Economy ==================
	"model4_name": "Model 4: National economy", 
	"model4_eq": "Y''(t) + αY'(t) + βY(t) = mlG",
	"model4_desc": ("The model simulates economic changes over time based on macroeconomic variables.<br>"
	"Y(t): National income.<br>"
	"Y'(t): Rate of change of income.<br>"
	"α: Coefficient related to propensity to consume/invest (α = m + ls - lma).<br>"
	"β: Investment response to income change (β = lms).<br>"
	"m: Government spending multiplier.<br>"
	"l: Parameter related to money demand.<br>"
	"a: Investment sensitivity to interest rate.<br>"
	"s: Money demand sensitivity to interest rate.<br>"
	"G: Government spending (constant)."),
	"model4_param_alpha": "Coefficient α", 
	"model4_param_beta": "Coefficient β", 
	"model4_param_a": "Parameter a",
	"model4_param_s": "Parameter s",
	"model4_param_m": "Multiplier m",
	"model4_param_G": "Spending G",
	"model4_param_l": "Parameter l",
	"model4_param_Y0": "Y(t₀)",
	"model4_param_dY0": "Y'(t₀)",
	"model4_param_t0": "Time t₀",
	"model4_param_t1": "Time t₁",
	# ================== MODEL 5: Pursuit Curve ==================
	"model5_name": "Model 5: Pursuit curve", 
	"model5_eq": "dx/dt = -v * x / √(x<sup>2</sup>+y<sup>2</sup>)<br>dy/dt = -v * y / √(x<sup>2</sup>+y<sup>2</sup>) - u",
	"model5_desc": ("Model to find the trajectory of a boat moving under the influence of a current.<br>"
	"(x(t), y(t)): Coordinates of the boat at time t.<br>"
	"v: Velocity of the boat (relative to water).<br>"
	"u: Velocity of the current (in the negative y-direction).<br>"
	"x<sub>0</sub>, y<sub>0</sub>: Initial coordinates of the boat.<br>"
	"t<sub>0</sub>, t<sub>1</sub>: Simulation time interval."),
	"model5_param_x0": "Initial x-coordinate (x₀)",
	"model5_param_y0": "Initial y-coordinate (y₀)",
	"model5_param_u": "Current velocity (u)",
	"model5_param_v": "Boat velocity (v)",
	"model5_param_t0": "Start time (t₀)",
	"model5_param_t1": "End time (t₁)",
	"model5_select_component": "Select component to display:",
	"model5_component_x": "X-Component",
	"model5_component_y": "Y-Component",
	
	# Screen 2
	"screen2_back_button": "Back to model selection", 
	"screen2_goto_screen3_button": "View dynamic simulation", 
	"screen2_goto_screen3_tooltip": "Switch to the dynamic simulation screen (if available)",
	"screen2_method_group": "1. Select main method",
	"screen2_method_ab": "Adams-Bashforth (AB)",
	"screen2_method_am": "Adams-Moulton (AM)",
	"screen2_details_group_ab": "2. Adams-Bashforth details", 
	"screen2_details_group_am": "2. Adams-Moulton details", 
	"screen2_steps_label": "Number of steps:",
	"screen2_select_all_steps_cb": "All",
	"screen2_step2": "2-step",
	"screen2_step3": "3-step",
	"screen2_step4": "4-step",
	"screen2_step5": "5-step",
	"param_placeholder": "Enter numeric value...",
	"screen2_h_label": "Step size (h):",
	"screen2_sim_toggle": "Show separate dynamic simulation window",
	"screen2_params_group": "3. Model input parameters", 
	"screen2_actions_group": "4. Execute and save", 
	"screen2_init_button": "🚀 Initialize and plot graphs", 
	"screen2_refresh_button": "🔄 Refresh parameters", 
	"screen2_show_data_button": "📊 View numerical data", 
	"screen2_save_button": "💾 Save plot images", 
	"screen2_plot_solution_title": "Solution plot", 
	"screen2_plot_error_title": "L∞ error plot", 
	"screen2_plot_order_title": "Convergence order plot", 
	"screen2_plot_t_axis": "Time (t)",
	"screen2_plot_value_axis": "Solution value",
	"screen2_plot_n_axis": "Number of subintervals (N)",
	"screen2_plot_h_axis": "Step size (h)",
	"screen2_plot_error_axis": "L∞ Error",
	"screen2_plot_log_h_axis": "log(h)",
	"screen2_plot_log_error_axis": "log(L∞ Error)",
	"screen2_plot_exact_label": "Exact solution / Reference",
	"screen2_plot_error_label_prefix": "Error ",
	"screen2_plot_order_data_label_suffix": " (data)",
	"screen2_plot_order_fit_label_suffix": " Fit: O(h<sup>{:.2f}</sup>)",
	"screen2_info_area_init": "Press 'Initialize and plot graphs' to start the simulation.",
	"screen2_info_area_running": "Processing, please wait...",
	"screen2_info_area_error": "Error: ",
	"screen2_info_area_no_results": "No simulation results to display.",
	"screen2_info_area_no_show_data": "No data available. Please run a simulation first.",
	"screen2_info_area_show_data_method": "Method:",
	"screen2_info_area_show_data_textCont1": "-step",
	"screen2_info_area_show_data_textCont2": "with h =",
	"screen2_info_area_show_data_order": "Estimated convergence order:",
	"screen2_info_area_show_data_points_header": "Table of error",
	"screen2_info_area_show_data_time": "t", 
	"screen2_info_area_show_data_approx": "Approximate", 
	"screen2_info_area_show_data_exact": "Exact",
	"screen2_info_area_show_data_error": "Error",
	"screen2_info_area_show_data_more": "(and other points...)",
	"screen2_info_area_show_data_no_points": "(No point data)",
	"screen2_info_area_refreshed": "Fields refreshed. Ready for new simulation.",
	"screen2_info_area_complete": "Completed!",
	"screen2_legend_title": "Legend",
	"screen2_legend_order_title": "Convergence order:", 
	"screen2_plot_value_axis_base": "Value",
	"screen2_plot_value_axis_x": "Value (X-component)", 
	"screen2_plot_value_axis_y": "Value (Y-component)", 
	"screen2_plot_error_axis_base": "L∞ Error",
	"screen2_plot_error_axis_x": "L∞ Error (X-component)", 
	"screen2_plot_error_axis_y": "L∞ Error (Y-component)", 
	"screen2_plot_log_error_axis_base": "log(L∞ Error)",
	"screen2_plot_log_error_axis_x": "log(L∞ Error X)",
	"screen2_plot_log_error_axis_y": "log(L∞ Error Y)",
	"screen2_info_area_show_data_approx_x": "Approximate",
	"screen2_info_area_show_data_approx_y": "Approximate",
	"screen2_info_area_show_data_exact_x": "Reference",
	"screen2_info_area_show_data_exact_y": "Reference",
	"screen2_info_area_show_data_error_x": "Error",
	"screen2_info_area_show_data_error_y": "Error",
	
	# Simulation Plot Window
	"sim_window_title": "Dynamic simulation plot window", 
	"sim_window_plot_title": "Solution simulation over time", 
	"sim_window_t_axis": "Time (t)",
	"sim_window_value_axis": "Solution Value",
	"sim_window_no_data": "No valid data to plot dynamically",
	
	# Messages
	"msg_error": "Error",
	"msg_warning": "Warning",
	"msg_info": "Information",
	"msg_select_method": "Please select a method.",
	"msg_select_step": "Please select at least one number of steps.",
	"msg_select_h": "Please select a step size (h).",
	"msg_invalid_h": "Invalid step size h: '{}'.",
	"msg_invalid_input_title": "Input error", 
	"msg_invalid_input_text": "Please check the input parameters.",
	"msg_invalid_param_value": "Parameter '{}' is not a valid number (received value: '{}').",
	"msg_missing_param_value": "Parameter '{}' cannot be empty.",
	"msg_model_error_title": "Model error", 
	"msg_model_no_ode": "Model '{}' does not have an ODE function defined.",
	"msg_param_error_title": "Parameter error", 
	"msg_missing_keys": "Missing required parameters: {}",
	"msg_missing_y0": "Missing initial condition (e.g., O₀, x₀, Y(t₀)).",
	"msg_param_value_error_title": "Parameter value error", 
	"msg_param_value_error_text": "Value error in parameter: {}",
	"msg_t_end_error": "End time t₁ must be greater than start time t₀.",
	"msg_unknown_error_title": "Unknown error", 
	"msg_unknown_error_prep": "An error occurred while preparing for the simulation: {}",
	"msg_internal_error_title": "Internal error", 
	"msg_internal_error_steps": "Invalid number of steps: {}",
	"msg_simulation_error_title": "Simulation error", 
	"msg_simulation_error_text": "An error occurred during the simulation process: {}",
	"msg_no_results_title": "No results", 
	"msg_no_results_text": "No valid simulation results were generated.",
	"msg_sim_window_no_data": "No data available to display in the dynamic simulation window.",
	"msg_show_data_no_data": "No simulation data has been generated yet.",
	"msg_save_no_plots": "No plots have been generated to save.",
	"msg_save_select_dir": "Please select a directory to save images",
	"msg_save_cancelled": "Directory selection was cancelled.",
	"msg_save_success_title": "Save successful", 
	"msg_save_success_text": "Saved {} image(s) to the directory:\n{}",
	"msg_save_error_title": "Save error", 
	"msg_save_error_text": "Saved {} image(s) but encountered some errors:\n{}",
	"msg_save_fail_title": "Save failed", 
	"msg_save_fail_text": "Could not save images due to an error:\n{}",
	"msg_saving_plot_error": "Error saving plot '{}': {}",
	"msg_skipping_save": "Skipping save for plot '{}' (no data).",
	
	#Screen 3
	"screen3_back_button": "Back to parameters", 
	"screen3_double_back_button": "Back to model selection", 
	"screen3_dyn_only_title": "Dynamic model simulation", 
	"screen3_dyn_back_tooltip": "Back to the input and static results screen",
	"screen3_settings_group_title": "Dynamic simulation settings", 
	"screen3_speed_label": "Playback speed:",
	"screen3_results_group_title": "Dynamic simulation information", 
	"screen3_unified_results_title": "Simulation information", 
	"screen3_stop_button": "⏹ Pause/Resume simulation", 
	"screen3_result_time": "Simulation time (t):",
	"screen3_result_c": "Growth constant (c):",
	"screen3_result_mass": "Cell count (or mass):",
	"screen3_legend_model2_cell": "Cell",
	"screen3_waiting_for_data": "Waiting for data or to start simulation...",
	"screen3_result_r_param": "Infection rate parameter (r):",
	"screen3_model3_simulation_time_label": "ABM simulation time (steps):",
	"screen3_actual_time": "Real time elapsed (seconds):",
	"screen3_total_pop": "Initial total population:",
	"screen3_infected_pop": "Number of infected individuals:",
	"screen3_susceptible_pop": "Number of susceptible individuals:",
	"screen3_legend_abm_susceptible": "Susceptible",
	"screen3_legend_abm_infected": "Infected",
	
	"screen3_sim_list_group_title": "Simulation List",
	"screen3_sim1_name_m5": "Simulation 1: Boat crossing river to a fixed target",
	"screen3_sim2_name_m5": "Simulation 2: Destroyer hunting submarine (In fog)",
	
	"screen3_info_m5_sim1_title": "Simulation 1 information (boat crossing)", 
	"screen3_m5_boat_speed": "Boat velocity (v):",
	"screen3_m5_water_speed": "Current velocity (u):",
	"screen3_m5_crossing_time": "Simulation time (t):",
	"screen3_m5_start_point_boat": "Boat's starting point:",
	"screen3_m5_boat_reaches_target": "Does the boat reach the target?",
	"screen3_m5_boat_final_pos": "Boat's final position:",
	"screen3_m5_determining_status": "Determining...",
	"answer_yes": "Yes",
	"answer_no": "No",
	
	"screen3_info_m5_sim2_title": "Simulation 2 information (destroyer vs. submarine)", 
	"screen3_m5_submarine_speed": "Submarine velocity (v<sub>target</sub>):",
	"screen3_m5_destroyer_speed": "Destroyer velocity (v<sub>pursuer</sub>):",
	"screen3_legend_m5s2_tn_avoid_radius": "Submarine radar zone", 
	"screen3_legend_m5s2_kt_radar_radius": "Destroyer radar zone", 
	"screen3_m5_submarine_trajectory": "Submarine base trajectory equation:",
	"screen3_m5_start_point_submarine": "Submarine's starting point:",
	"screen3_m5_start_point_destroyer": "Destroyer's starting point:",
	"screen3_m5_destroyer_catches_submarine": "Does the destroyer catch the submarine?",
	"screen3_m5_catch_point": "Catch point (coordinates):",
	"screen3_m5_catch_time": "Simulation time:", 
	"screen3_model5_not_implemented_msg": "Simulation for this scenario is not yet implemented.",
	
	"screen3_model2_anim_plot_title": "Dynamic simulation: Cell growth", 
	"screen3_abm_anim_plot_title": "Dynamic simulation: Epidemic spread (ABM)", 
	
	"screen3_model5_plot_title_sim1": "Dynamic simulation: Boat crossing river", 
	"screen3_model5_plot_title_sim2": "Dynamic simulation: Destroyer - Submarine", 
	
	"screen3_model5_plot_xlabel_sim1": "X Position (m)",
	"screen3_model5_plot_ylabel_sim1": "Y Position (m)",
	"screen3_legend_m5s1_path": "Boat trajectory", 
	"screen3_legend_m5s1_boat": "Boat",
	"screen3_legend_m5s1_water_current": "Water current direction", 
	
	"screen3_model5_plot_xlabel_sim2": "X Position (m)",
	"screen3_model5_plot_ylabel_sim2": "Y Position (m)",
	"screen3_legend_m5s2_submarine": "Submarine",
	"screen3_legend_m5s2_destroyer": "Destroyer",
	"screen3_legend_m5s2_path_submarine": "Submarine trajectory", 
	"screen3_legend_m5s2_path_destroyer": "Destroyer trajectory", 
	"screen3_legend_m5s2_catch_point": "Catch point",
	"screen3_m5_time_when_caught_label": "Time to catch submarine:",
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
		
MODELS_DATA = {
    #Model 1: Energy demand
    LANG_VI["model1_name"]: {
        "id": "model1",
        "equation_key": "model1_eq",
        "description_key": "model1_desc",
        "param_keys_vi": [LANG_VI["model1_param1"], LANG_VI["model1_param2"], LANG_VI["model1_param3"], LANG_VI["model1_param4"]],
        "param_keys_en": [LANG_EN["model1_param1"], LANG_EN["model1_param2"], LANG_EN["model1_param3"], LANG_EN["model1_param4"]],
        "internal_param_keys": ["O₀", "k", "t₀", "t₁"],
        "ode_func": lambda k: (lambda t, y: k * y),
        "exact_func": lambda O0, k, t0: (lambda t: O0 * np.exp(k * (np.asarray(t) - t0))),
    },
    #Model 2: Cell growth
    LANG_VI["model2_name"]: {
        "id": "model2",
        "equation_key": "model2_eq",
        "description_key": "model2_desc",
        "param_keys_vi": [LANG_VI["model2_param1"], LANG_VI["model2_param3"], LANG_VI["model2_param4"]],
        "param_keys_en": [LANG_EN["model2_param1"], LANG_EN["model2_param3"], LANG_EN["model2_param4"]],
        "internal_param_keys": ["x₀", "t₀", "t₁"],
        "ode_func": lambda c: (lambda t, y: c * (y**(2.0/3.0) + 1e-15)),
        "exact_func": lambda x0, c, t0: (lambda t: (x0**(1.0/3.0) + c * (np.asarray(t) - t0) / 3.0)**3),
    },
    #Model 3: Spread of epidemicepidemic
    LANG_VI["model3_name"]: {
        "id": "model3", 
        "can_run_abm_on_screen3": True,
        "equation_key": "model3_eq",
        "description_key": "model3_desc",
        "param_keys_vi": [LANG_VI["model3_param2"], LANG_VI["model3_param4"], LANG_VI["model3_param5"]],
        "param_keys_en": [LANG_EN["model3_param2"], LANG_EN["model3_param4"], LANG_EN["model3_param5"]],
        "internal_param_keys": ["n", "t₀", "t₁"], 
        "ode_func": lambda r, n_initial: (lambda t, y: -r * y * (n_initial + 1.0 - y)),
        "exact_func": lambda n_initial, r, t0: (
            lambda t: (n_initial * (n_initial + 1.0) * np.exp(-r * (n_initial + 1.0) * (np.asarray(t) - t0))) / \
                      (1.0 + n_initial * np.exp(-r * (n_initial + 1.0) * (np.asarray(t) - t0))) if n_initial > 0 else
            (lambda t: np.zeros_like(np.asarray(t))) 
        ),
        "abm_defaults": {
            "initial_infected": 1,
            "room_dimension": ABM_ROOM_DIMENSION_DEFAULT, 
            "r_to_ptrans_factor": 5000,
            "ptrans_min": ABM_PTRANS_MIN, 
            "ptrans_max": ABM_PTRANS_MAX, 
            "base_agent_speed": 0.04,
            "speed_scaling_factor": 0.5,
            "min_agent_speed": 0.02,
            "max_agent_speed": 0.20,

            "base_contact_radius": 0.5,
            "radius_scaling_factor": 3.0,
            "min_contact_radius": 0.3,
            "max_contact_radius": 1.5,
            "seconds_per_step": 0.1,

            "max_steps": ABM_MAX_STEPS_DEFAULT, 
            "interval_ms": ABM_INTERVAL_DEFAULT, 
            "display_max_total": MAX_TOTAL_AGENTS_FOR_FULL_DISPLAY, 
            "display_sample_size": SAMPLE_SIZE_FOR_LARGE_POPULATION 
        }
    },
    #Model 4: National economy
    LANG_VI["model4_name"]: {
        "id": "model4",
        "is_system": True,
        "equation_key": "model4_eq",
        "description_key": "model4_desc",
        "param_keys_vi": [
            LANG_VI["model4_param_m"], LANG_VI["model4_param_l"],
            LANG_VI["model4_param_a"], LANG_VI["model4_param_s"], 
            LANG_VI["model4_param_G"],
            LANG_VI["model4_param_alpha"], LANG_VI["model4_param_beta"], 
            LANG_VI["model4_param_dY0"], LANG_VI["model4_param_Y0"], 
            LANG_VI["model4_param_t0"], LANG_VI["model4_param_t1"]
        ],
        "param_keys_en": [
            LANG_EN["model4_param_m"], LANG_EN["model4_param_l"],
            LANG_EN["model4_param_a"], LANG_EN["model4_param_s"], 
            LANG_EN["model4_param_G"],
            LANG_EN["model4_param_alpha"], LANG_EN["model4_param_beta"], 
            LANG_EN["model4_param_dY0"], LANG_EN["model4_param_Y0"],
            LANG_EN["model4_param_t0"], LANG_EN["model4_param_t1"]
        ],
        "internal_param_keys": ["m", "l", "a", "s", "G", "Y0", "dY0", "t₀", "t₁"], 
        "ode_func": lambda alpha, beta, m, G, l: (
            lambda t, u1, u2: np.array([u2, m * l * G - alpha * u2 - beta * u1])
        ),
        "exact_func": lambda alpha, beta, m, G, l, n, k, t0: (
            lambda t_arr: _model4_exact_solution(alpha, beta, m, G, l, n, k, t0, t_arr)
        ),
    },
    #Model 5: Pursuit curve
    LANG_VI["model5_name"]: {
        "id": "model5",
        "is_system": True,                 
        "uses_rk5_reference": True,      
        "equation_key": "model5_eq",
        "description_key": "model5_desc",
        "param_keys_vi": [
            LANG_VI["model5_param_x0"], LANG_VI["model5_param_y0"],
            LANG_VI["model5_param_u"], LANG_VI["model5_param_v"] ,
            LANG_VI["model5_param_t0"], LANG_VI["model5_param_t1"],
        ],
        "param_keys_en": [
            LANG_EN["model5_param_x0"], LANG_EN["model5_param_y0"],
             LANG_EN["model5_param_u"], LANG_EN["model5_param_v"],
            LANG_EN["model5_param_t0"], LANG_EN["model5_param_t1"],
        ],
        "internal_param_keys": ["x0", "y0", "u", "v", "t₀", "t₁"], 
        "ode_func": lambda u_param, v_param: (
            lambda t, x, y: _model5_ode_system(t, x, y, u_param, v_param)
        ),
        "exact_func": None,
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
# --- 4. CẤU TRÚC CHÍNH CỦA ỨNG DỤNG STREAMLIT ---

# Khởi tạo st.session_state để lưu trạng thái
def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'
    if 'lang' not in st.session_state:
        st.session_state.lang = 'vi'
    if 'translations' not in st.session_state:
        st.session_state.translations = LANG_VI
    if 'selected_model_key' not in st.session_state:
        st.session_state.selected_model_key = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = {}
    if 'validated_params' not in st.session_state:
        st.session_state.validated_params = {}

# Hàm tiện ích để dịch văn bản
def tr(key):
    return st.session_state.translations.get(key, key)

# Hàm chính để điều hướng giữa các trang
def main():
    # Bước 1: Khởi tạo session_state trước tiên để đảm bảo nó tồn tại
    initialize_session_state()

    # Bước 2: Cấu hình trang. Bây giờ hàm tr() đã có thể truy cập session_state an toàn
    st.set_page_config(layout="wide", page_title=tr("app_title"))	
  	
    # Bước 3: Chạy logic điều hướng trang như bình thường
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
    # Sử dụng markdown và st.columns để tạo layout
    st.markdown(f"<h2 style='text-align: center; color: #000080;'>{tr('welcome_uni')}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: #000080;'>{tr('welcome_faculty')}</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown(f"<h1 style='text-align: center; color: #990000;'>{tr('welcome_project_title').replace('\\n', '<br>')}</h1>", unsafe_allow_html=True)
    st.write("") # Thêm khoảng trống

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"**{tr('welcome_authors_title')}**")
        st.markdown(tr('welcome_authors_names'))
        st.markdown(f"**{tr('welcome_advisors_title')}**")
        st.markdown(tr('welcome_advisor1'))
        st.markdown(tr('welcome_advisor2'))
        st.write("")
    
        # Language selector
        lang_options = {tr('lang_vi'): 'vi', tr('lang_en'): 'en'}
        selected_lang_display = st.radio(
            "Chọn ngôn ngữ / Select Language:",
            options=lang_options.keys(),
            horizontal=True,
            index=0 if st.session_state.lang == 'vi' else 1
        )

        selected_lang_code = lang_options[selected_lang_display]
        if selected_lang_code != st.session_state.lang:
            st.session_state.lang = selected_lang_code
            st.session_state.translations = LANG_VI if selected_lang_code == 'vi' else LANG_EN
            st.rerun()

    st.write("")
    col_logo1, col_title, col_logo2 = st.columns([1, 4, 1])

    with col_logo1:
        # Highlight: Cập nhật đường dẫn và hiển thị logo TDTU
        logo_tdtu_path = os.path.join(FIG_FOLDER, "logotdtu1.png")
        if os.path.exists(logo_tdtu_path):
            st.image(logo_tdtu_path, width=150)
        else:
            st.write("[TDTU Logo Error]")

    with col_title:
        st.markdown(f"<h2 style='text-align: center; color: #000080;'>{tr('welcome_uni')}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #000080;'>{tr('welcome_faculty')}</h2>", unsafe_allow_html=True)

    with col_logo2:
        # Highlight: Cập nhật đường dẫn và hiển thị logo Khoa
        logo_faculty_path = os.path.join(FIG_FOLDER, "logokhoa1@2.png")
        if os.path.exists(logo_faculty_path):
            st.image(logo_faculty_path, width=100)
        else:
            st.write("[Faculty Logo Error]")

    # Nút bắt đầu
    col1_btn, col2_btn, col3_btn = st.columns([2,1,2])
    with col2_btn:
        if st.button(tr('start_button'), use_container_width=True):
            st.session_state.page = 'model_selection'
            st.rerun()

# --- Thay thế hàm show_model_selection_page cũ ---
def show_model_selection_page():
    st.title(tr('screen1_title'))
    
    # Nút quay lại
    if st.button(f"ᐊ {tr('welcome_project_title').splitlines()[0]}"):
        st.session_state.page = 'welcome'
        st.rerun()

    # Model selector
    model_display_names = [tr(f"{data['id']}_name") for data in MODELS_DATA.values()]
    # Dùng key tiếng Việt làm key duy nhất để tra cứu
    model_vi_keys = list(MODELS_DATA.keys())
    
    # Tìm index của model đang được chọn (nếu có)
    current_selection_index = 0
    if st.session_state.selected_model_key in model_vi_keys:
        current_selection_index = model_vi_keys.index(st.session_state.selected_model_key)

    selected_model_display_name = st.selectbox(
        label=" ", # Ẩn label
        options=model_display_names,
        index=current_selection_index
    )

    # Tìm lại key tiếng Việt từ tên hiển thị đã được dịch
    selected_model_index = model_display_names.index(selected_model_display_name)
    selected_key = model_vi_keys[selected_model_index]
    st.session_state.selected_model_key = selected_key
    
    model_data = MODELS_DATA[selected_key]
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(tr('screen1_model_info_group_title'))
        st.markdown(f"**{tr('screen1_equation_label')}**")
        st.markdown(tr(model_data['equation_key']), unsafe_allow_html=True)
        
        st.markdown(f"**{tr('screen1_description_label')}**")
        st.markdown(tr(model_data['description_key']), unsafe_allow_html=True)

    with col2:
        st.subheader(tr('screen1_model_application_group_title'))
        model_id = model_data.get("id")
        lang_suffix = "Vie" if st.session_state.lang == 'vi' else "Eng"
        image_filename = f"model_{model_id[5:]}_{lang_suffix}.png"
        image_path = os.path.join(FIG_FOLDER, image_filename)
        
        if os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning(f"Không tìm thấy ảnh: {image_filename}")
            
    st.write("---")
    
    if st.button(tr('screen1_continue_button'), type="primary"):
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
    Hàm này chuẩn bị mọi thứ cần thiết để chạy mô phỏng.
    Nó thay thế cho logic bên trong _prepare_simulation_functions từ code gốc.
    """
    try:
        ode_gen = model_data.get("ode_func")
        exact_gen = model_data.get("exact_func")
        model_id = model_data.get("id")
        if not callable(ode_gen):
            raise ValueError(tr("msg_model_no_ode").format(tr(f"{model_id}_name")))
        
        t_start = input_params['t₀']
        t_end = input_params['t₁']
        y0 = None

        if model_id == "model4":
            if 'Y0' not in input_params or 'dY0' not in input_params:
                raise ValueError(tr("msg_missing_y0"))
            y0 = [input_params['Y0'], input_params['dY0']]
        elif model_id == "model5":
            if 'x0' not in input_params or 'y0' not in input_params:
                raise ValueError(tr("msg_missing_y0"))
            y0 = [input_params['x0'], input_params['y0']]
        else:
            y0_key_map = {'model1': 'O₀', 'model2': 'x₀', 'model3': 'n'}
            y0_key = y0_key_map.get(model_id)
            if y0_key is None or y0_key not in input_params:
                raise ValueError(tr("msg_missing_y0"))
            y0 = input_params[y0_key]
        
        ode_func = None
        exact_callable = None
        
        # Logic tính toán tham số đặc biệt cho từng model
        if model_id == "model1":
            k = input_params['k']
            ode_func = ode_gen(k)
            exact_callable = exact_gen(y0, k, t_start) if callable(exact_gen) else None
        elif model_id == "model2":
            number_of_double_times = 5.0 if selected_method_short == "Bashforth" else 2.0
            denominator_b = input_params['t₁']
            if denominator_b <= 1e-9 or input_params['x₀'] < 0:
                raise ValueError("t₁ phải > 0 và x₀ >= 0 cho Model 2")
            x0_cbrt_safe = (input_params['x₀'] + 1e-15)**(1.0/3.0)
            doubling_factor_N_cbrt = (2.0**number_of_double_times)**(1.0/3.0)
            lower_c = 3.0 * (doubling_factor_N_cbrt - 1.0) * x0_cbrt_safe / denominator_b
            doubling_factor_Nplus1_cbrt = (2.0**(number_of_double_times + 1.0))**(1.0/3.0)
            upper_c = 3.0 * (doubling_factor_Nplus1_cbrt - 1.0) * x0_cbrt_safe / denominator_b
            calculated_c = random.uniform(min(lower_c, upper_c), max(lower_c, upper_c))
            
            # Lưu lại để hiển thị trên UI
            st.session_state.last_calculated_c = calculated_c
            
            ode_func = ode_gen(calculated_c)
            if callable(exact_gen):
                exact_callable = exact_gen(y0, calculated_c, t_start)
        elif model_id == "model3":
            n_initial = y0
            calculated_r = _predict_r_for_model3(t_start, t_end, n_initial)
            if calculated_r is None:
                raise ValueError("Không thể tính toán tham số 'r' cho Model 3.")
            
            # Lưu lại để hiển thị
            st.session_state.last_calculated_r = calculated_r
            
            ode_func = ode_gen(calculated_r, n_initial)
            if callable(exact_gen):
                exact_callable = exact_gen(n_initial, calculated_r, t_start)
        elif model_id == "model4":
            m, l, a, s, G = input_params['m'], input_params['l'], input_params['a'], input_params['s'], input_params['G']
            Y0_val, dY0_val = y0[0], y0[1]
            
            alpha_calculated = m + l * s - l * m * a
            beta_calculated = l * m * s
            
            # Lưu lại để hiển thị
            st.session_state.last_calculated_alpha = alpha_calculated
            st.session_state.last_calculated_beta = beta_calculated
            
            ode_func = ode_gen(alpha_calculated, beta_calculated, m, G, l)
            if callable(exact_gen):
                exact_callable = exact_gen(alpha_calculated, beta_calculated, m, G, l, Y0_val, dY0_val, t_start)
        elif model_id == "model5":
            u_param = input_params['u']
            v_param = input_params['v']
            ode_func = ode_gen(u_param, v_param)
            exact_callable = None # Model 5 không có nghiệm giải tích
        else:
            raise NotImplementedError(f"Logic chuẩn bị cho model '{model_id}' chưa được triển khai.")

        if not callable(ode_func):
            raise RuntimeError(f"Lỗi nội bộ: Hàm ODE không được tạo cho model '{model_id}'.")
        
        return True, (ode_func, exact_callable, y0, t_start, t_end)

    except Exception as e:
        st.error(tr("msg_unknown_error_prep").format(e))
        return False, None

def _perform_single_simulation(model_data, ode_func, exact_sol_func, y0, t_start, t_end, method_short, steps_int, h_target, selected_component='x'):
    """
    Thực hiện một lần mô phỏng đầy đủ, bao gồm cả tính toán bậc hội tụ.
    Đây là phiên bản Streamlit của hàm cùng tên trong code gốc.
    """
    is_system = model_data.get("is_system", False)
    uses_rk5_reference = model_data.get("uses_rk5_reference", False)
    model_id = model_data.get("id")

    # --- Chọn hàm solver ---
    method_map_single = {"Bashforth": {2: AB2, 3: AB3, 4: AB4, 5: AB5}, "Moulton": {2: AM2, 3: AM3, 4: AM4}}
    method_map_system = {"Bashforth": {2: AB2_system, 3: AB3_system, 4: AB4_system, 5: AB5_system}, "Moulton": {2: AM2_system, 3: AM3_system, 4: AM4_system}}
    method_map_model5 = {
        "Bashforth": {2: AB2_original_system_M5, 3: AB3_original_system_M5, 4: AB4_original_system_M5, 5: AB5_original_system_M5},
        "Moulton": {2: AM2_original_system_M5, 3: AM3_original_system_M5, 4: AM4_original_system_M5}
    }
    
    current_map = method_map_single
    if model_id == "model5": current_map = method_map_model5
    elif is_system: current_map = method_map_system
    
    method_func = current_map.get(method_short, {}).get(steps_int)
    if method_func is None:
        st.error(f"Solver không tồn tại cho {method_short} {steps_int} bước.")
        return None

    # --- Logic tính toán cho đồ thị và bậc hội tụ (giữ nguyên từ code gốc) ---
    # ... (Toàn bộ logic từ _perform_single_simulation gốc sẽ được sao chép vào đây) ...
    # ... Bắt đầu sao chép ...
    interval_length = t_end - t_start
    if interval_length <= 1e-9: return None
    min_n_required = max(steps_int, 2)
    n_plot = max(int(np.ceil(interval_length / h_target)), min_n_required if model_id != "model5" else 5)
    if uses_rk5_reference: n_plot = max(n_plot, 50)
    h_actual_plot = interval_length / n_plot
    t_plot = np.linspace(t_start, t_end, n_plot + 1)
    y_approx_plot_u1, y_exact_plot_u1 = None, None
    y_approx_plot_all_components, y_exact_plot_all_components = None, None
    
    rk5_ref_for_screen2 = RK5_original_system_M5 if model_id == "model5" else None

    if model_id == "model5" and uses_rk5_reference and rk5_ref_for_screen2:
        try:
            rk5_ref_x_plot, rk5_ref_y_plot = rk5_ref_for_screen2(ode_func, t_plot, y0[0], y0[1])
            y_exact_plot_all_components = [np.asarray(rk5_ref_x_plot), np.asarray(rk5_ref_y_plot)]
            y_exact_plot_u1 = y_exact_plot_all_components[0 if selected_component == 'x' else 1]
        except Exception as e_rk5_plot:
            print(f"    Error calculating RK5 reference for Model 5 main plot: {e_rk5_plot}")
            return None
    elif exact_sol_func and len(t_plot) > 0:
        if is_system:
            exact_tuple = exact_sol_func(t_plot)
            if exact_tuple is not None and len(exact_tuple) == 2:
                y_exact_plot_all_components = [np.asarray(exact_tuple[0]), np.asarray(exact_tuple[1])]
                y_exact_plot_u1 = y_exact_plot_all_components[0]
        else:
            y_exact_plot_u1 = np.asarray(exact_sol_func(t_plot))
    try:
        if is_system:
            u1_plot, u2_plot = method_func(ode_func, t_plot, y0[0], y0[1])
            y_approx_plot_all_components = [np.asarray(u1_plot), np.asarray(u2_plot)]
            y_approx_plot_u1 = y_approx_plot_all_components[0 if selected_component == 'x' else 1]
        else:
            y_plot = method_func(ode_func, t_plot, y0)
            y_approx_plot_u1 = np.asarray(y_plot)
        if y_approx_plot_u1 is not None:
            min_len_plot = len(y_approx_plot_u1)
            if y_exact_plot_u1 is not None: min_len_plot = min(min_len_plot, len(y_exact_plot_u1))
            min_len_plot = min(min_len_plot, len(t_plot))
            t_plot = t_plot[:min_len_plot]
            if y_approx_plot_all_components:
                y_approx_plot_all_components[0] = y_approx_plot_all_components[0][:min_len_plot]
                y_approx_plot_all_components[1] = y_approx_plot_all_components[1][:min_len_plot]
                y_approx_plot_u1 = y_approx_plot_all_components[0 if selected_component == 'x' else 1]
            elif y_approx_plot_u1 is not None: y_approx_plot_u1 = y_approx_plot_u1[:min_len_plot]
            if y_exact_plot_all_components:
                y_exact_plot_all_components[0] = y_exact_plot_all_components[0][:min_len_plot]
                y_exact_plot_all_components[1] = y_exact_plot_all_components[1][:min_len_plot]
                y_exact_plot_u1 = y_exact_plot_all_components[0 if selected_component == 'x' else 1]
            elif y_exact_plot_u1 is not None: y_exact_plot_u1 = y_exact_plot_u1[:min_len_plot]
    except Exception as e_plot_approx:
        print(f"    Error calculating APPROXIMATE solution for main plot (N={n_plot}) using {method_func.__name__}: {e_plot_approx}")
        t_plot = np.array([])
        y_approx_plot_u1, y_exact_plot_u1 = None, None
        y_approx_plot_all_components, y_exact_plot_all_components = None, None
    errors_convergence = []
    h_values_for_loglog_list = []
    n_values_plotted = []
    log_h_conv, log_err_conv = [], []
    slope = np.nan
    interval_n_base = max(1, int(np.ceil(interval_length)))
    n_start_conv, n_end_conv = 0, 0
    num_points_convergence = 10
    if model_id == "model1": n_start_conv, n_end_conv = max(5, 2*interval_n_base), max(20, 10*interval_n_base)
    elif model_id == "model2": n_start_conv, n_end_conv = max(5,interval_n_base), max(10,interval_n_base*2)
    elif model_id == "model3": n_start_conv = interval_n_base; n_end_conv = 3 * interval_n_base
    elif model_id == "model4": n_start_conv, n_end_conv = max(10, 10*interval_n_base), max(30, 30*interval_n_base)
    elif model_id == "model5":
        n_start_conv = 2000
        n_end_conv = 10000
        num_points_convergence = 8
    else: n_start_conv, n_end_conv = 10, 100
    n_values_for_conv_loop = np.array([], dtype=int)
    if model_id in ["model2", "model3"]:
        if n_start_conv <= 0: n_start_conv = 1
        if n_end_conv <= n_start_conv: n_end_conv = n_start_conv + max(1, num_points_convergence -1)
        n_values_for_conv_loop = np.arange(n_start_conv, n_end_conv +1 , 1, dtype=int)
    elif model_id == "model5":
        if n_start_conv > n_end_conv : n_start_conv, n_end_conv = n_end_conv, n_start_conv
        if n_start_conv <= 0: n_start_conv = max(1, min_n_required)
        if n_end_conv <= n_start_conv: n_end_conv = n_start_conv + num_points_convergence
        n_values_for_conv_loop = np.linspace(n_start_conv, n_end_conv, num_points_convergence, dtype=int)
    else:
        if n_start_conv > n_end_conv: n_start_conv, n_end_conv = n_end_conv, n_start_conv
        if n_start_conv <= 0 and n_end_conv > 0: n_start_conv = 1
        if n_start_conv == n_end_conv and n_start_conv <= 0: return None
        if n_end_conv >= n_start_conv > 0:
            target_points_conv = 20
            range_n_conv = n_end_conv - n_start_conv
            step_n_conv = 1
            if range_n_conv > 0 and target_points_conv > 0:
                 step_n_conv = max(1, int(np.ceil(range_n_conv / target_points_conv)))
            n_values_for_conv_loop = np.arange(n_start_conv, n_end_conv + 1, step_n_conv, dtype=int)
    if len(n_values_for_conv_loop) == 0:
        if n_start_conv > 0: n_values_for_conv_loop = np.array([n_start_conv],dtype=int)
        else: print("Error: n_values_for_conv_loop is empty after generation."); return None
    n_values_filtered_conv = np.unique(n_values_for_conv_loop[n_values_for_conv_loop >= min_n_required])
    if len(n_values_filtered_conv) < 2:
        print(f"    Warning: Not enough N values ({len(n_values_filtered_conv)}) for convergence plot after filtering >= {min_n_required}.")
    else:
        print(f"    Convergence loop N values (original, filtered): Range [{n_values_filtered_conv[0]}, {n_values_filtered_conv[-1]}], Points: {len(n_values_filtered_conv)}")
        for n_conv_original in n_values_filtered_conv:
            n_eff_conv_sim = n_conv_original
            h_for_logplot_conv = 0.0
            if model_id == 'model2': n_eff_conv_sim = n_conv_original * 2
            elif model_id == 'model3': n_eff_conv_sim = n_conv_original * 10
            elif model_id == 'model4': n_eff_conv_sim = n_conv_original * 2
            if n_eff_conv_sim > 0: h_for_logplot_conv = interval_length / n_eff_conv_sim
            else: continue
            n_eff_conv_sim = max(n_eff_conv_sim, min_n_required)
            t_conv_loop = np.linspace(t_start, t_end, n_eff_conv_sim + 1)
            if len(t_conv_loop) < steps_int + 1: continue
            try:
                if is_system:
                    u1_approx_conv_loop, u2_approx_conv_loop = method_func(ode_func, t_conv_loop, y0[0], y0[1])
                    y_approx_conv_u1_selected_loop = u1_approx_conv_loop if selected_component == 'x' else u2_approx_conv_loop
                else:
                    y_approx_conv_loop = method_func(ode_func, t_conv_loop, y0)
                    y_approx_conv_u1_selected_loop = y_approx_conv_loop
                y_exact_conv_u1_selected_loop = None
                if model_id == "model5" and uses_rk5_reference and rk5_ref_for_screen2:
                    rk5_ref_x_conv_loop, rk5_ref_y_conv_loop = rk5_ref_for_screen2(ode_func, t_conv_loop, y0[0], y0[1])
                    y_exact_conv_u1_selected_loop = rk5_ref_x_conv_loop if selected_component == 'x' else rk5_ref_y_conv_loop
                elif exact_sol_func and len(t_conv_loop) > 0:
                    if is_system:
                        exact_tuple_loop = exact_sol_func(t_conv_loop)
                        if exact_tuple_loop is not None and len(exact_tuple_loop) == 2:
                            y_exact_conv_u1_selected_loop = exact_tuple_loop[0]
                    else:
                        y_exact_conv_u1_selected_loop = exact_sol_func(t_conv_loop)
                if y_approx_conv_u1_selected_loop is None or y_exact_conv_u1_selected_loop is None : continue
                y_approx_conv_u1_selected_loop = np.asarray(y_approx_conv_u1_selected_loop)
                y_exact_conv_u1_selected_loop = np.asarray(y_exact_conv_u1_selected_loop)
                approx_len_conv = len(y_approx_conv_u1_selected_loop)
                exact_len_conv = len(y_exact_conv_u1_selected_loop)
                min_len_conv_loop = min(approx_len_conv, exact_len_conv)
                if min_len_conv_loop < 2: continue
                approx_conv = y_approx_conv_u1_selected_loop[:min_len_conv_loop]
                exact_conv = y_exact_conv_u1_selected_loop[:min_len_conv_loop]
                error_conv = np.linalg.norm(exact_conv - approx_conv, np.inf)
                if np.isfinite(error_conv) and error_conv > 1e-16 and h_for_logplot_conv > 1e-16:
                    errors_convergence.append(error_conv)
                    n_values_plotted.append(n_conv_original)
                    h_values_for_loglog_list.append(h_for_logplot_conv)
                else:
                    print(f"    Skipping error point for N_orig={n_conv_original}: error={error_conv}, h_for_logplot={h_for_logplot_conv}")
            except Exception as e_conv_loop:
                 print(f"    Error during convergence step N_orig={n_conv_original}, N_eff={n_eff_conv_sim}: {e_conv_loop}")
    valid_h_for_loglog = np.array(h_values_for_loglog_list)
    if len(errors_convergence) >= 2 and len(valid_h_for_loglog) == len(errors_convergence):
            try:
                h_log_array = valid_h_for_loglog
                err_log_array = np.array(errors_convergence)
                valid_indices_log = np.where((h_log_array > 1e-16) & (err_log_array > 1e-16) & np.isfinite(h_log_array) & np.isfinite(err_log_array))[0]
                if len(valid_indices_log) >= 2:
                    log_h_conv = np.log(h_log_array[valid_indices_log])
                    log_err_conv = np.log(err_log_array[valid_indices_log])
                    if len(log_h_conv) >= 2:
                        coeffs = np.polyfit(log_h_conv, log_err_conv, 1)
                        slope = coeffs[0]
                        print(f"    Convergence analysis: Found {len(log_h_conv)} valid log points. Est. order: {slope:.3f}")
                    else:
                        print(f"    Warning: Not enough finite log values for polyfit ({len(log_h_conv)} points).")
                        slope = np.nan
                else:
                    print("    Warning: Less than 2 valid points after filtering for log calculation.")
                    slope = np.nan
            except Exception as fit_e:
                print(f"    Error during polyfit: {fit_e}")
                slope = np.nan
    else:
        print(f"    Warning: Not enough points for convergence analysis (errors: {len(errors_convergence)}, valid_h: {len(valid_h_for_loglog)}).")
        slope = np.nan
    # ... Kết thúc sao chép ...
    
    return {
        "t_plot": np.asarray(t_plot),
        "exact_sol_plot": np.asarray(y_exact_plot_u1) if y_exact_plot_u1 is not None else None,
        "approx_sol_plot": np.asarray(y_approx_plot_u1) if y_approx_plot_u1 is not None else None,
        "exact_sol_plot_all_components": y_exact_plot_all_components,
        "approx_sol_plot_all_components": y_approx_plot_all_components,
        "h_values_for_loglog": valid_h_for_loglog,
        "errors_convergence": np.array(errors_convergence),
        "log_h_convergence": log_h_conv,
        "log_error_convergence": log_err_conv,
        "order_slope": slope,
        "n_values_convergence": np.array(n_values_plotted),
        "h_actual_plot": h_actual_plot,
        "n_plot": n_plot,
        "selected_component": selected_component
    }


# --- HÀM CHÍNH CỦA TRANG MÔ PHỎNG (THAY THẾ HÀM CŨ) ---
def show_simulation_page():
    if not st.session_state.selected_model_key:
        st.warning("Vui lòng chọn một mô hình từ trang trước.")
        if st.button("Quay lại trang chọn mô hình"):
            st.session_state.page = 'model_selection'
            st.rerun()
        return

    model_data = MODELS_DATA[st.session_state.selected_model_key]
    model_id = model_data.get("id", "")
    model_name_tr = tr(f"{model_id}_name")

    # --- Header của trang ---
    col_h1, col_h2, col_h3 = st.columns([1, 4, 1])
    with col_h1:
        if st.button(f"ᐊ {tr('screen1_title')}"):
            st.session_state.page = 'model_selection'
            st.session_state.simulation_results = {}
            st.session_state.validated_params = {}
            st.rerun()
    with col_h2:
        st.title(model_name_tr)
    with col_h3:
        show_dynamic_button = model_data.get("can_run_abm_on_screen3", False) or model_id in ["model2", "model5"]
        if show_dynamic_button and st.session_state.simulation_results:
            if st.button(tr('screen2_goto_screen3_button')):
                st.session_state.page = 'dynamic_simulation'
                st.rerun()

    st.markdown("---")
    
    col_controls, col_display = st.columns([1, 2])

    with col_controls:
        # 1. Chọn phương pháp
        st.subheader(tr('screen2_method_group'))
        method_options = {tr('screen2_method_ab'): "Bashforth", tr('screen2_method_am'): "Moulton"}
        selected_method_display = st.radio("Phương pháp", options=method_options.keys(), label_visibility="collapsed", horizontal=True, key=f"method_{model_id}")
        selected_method_short = method_options[selected_method_display]

        # 2. Chi tiết
        details_title = tr('screen2_details_group_ab') if selected_method_short == 'Bashforth' else tr('screen2_details_group_am')
        st.subheader(details_title)

        step_options = {tr('screen2_step2'): 2, tr('screen2_step3'): 3, tr('screen2_step4'): 4}
        if selected_method_short == 'Bashforth' and model_id != "model5":
            step_options[tr('screen2_step5')] = 5
        selected_steps_display = st.multiselect(tr('screen2_steps_label'), options=step_options.keys(), default=list(step_options.keys())[2] if len(step_options) > 2 else list(step_options.keys())[0], key=f"steps_{model_id}_{selected_method_short}")
        selected_steps_int = [step_options[s] for s in selected_steps_display]
        
        h_values = ["0.1", "0.05", "0.01", "0.005", "0.001"]
        selected_h_str = st.radio(tr('screen2_h_label'), options=h_values, index=2, horizontal=True, key=f"h_{model_id}")
        h_float = float(selected_h_str)

        # 3. Tham số
        st.subheader(tr('screen2_params_group'))
        param_inputs = {}
        param_labels_key = f"param_keys_{st.session_state.lang}"
        all_param_labels = model_data.get(param_labels_key, model_data.get("param_keys_vi", []))
        internal_keys = model_data.get("internal_param_keys", [])

        default_values = {'t₀': 0.0, 't₁': 10.0, 'O₀': 1.0, 'k': 0.5, 'x₀': 1.0, 'n': 10.0,
                          'm': 0.5, 'l': 0.2, 'a': 0.1, 's': 0.25, 'G': 20.0, 'Y0': 100.0, 'dY0': 1.0,
                          'x0': 10.0, 'y0': 0.0, 'u': 1.0, 'v': 2.0}

        if model_id == "model4":
            cols_m4 = st.columns(2)
            for i, key in enumerate(internal_keys):
                label = tr(f"model4_param_{key}") if key in ['m', 'l', 'a', 's', 'G', 'Y0', 'dY0', 't₀', 't₁'] else key
                with cols_m4[i%2]:
                    param_inputs[key] = st.number_input(label, value=default_values.get(key, 0.0), format="%.4f", key=f"param_{model_id}_{key}")
        elif model_id == "model5":
            cols_m5 = st.columns(2)
            for i, key in enumerate(internal_keys):
                label = all_param_labels[i]
                with cols_m5[i%2]:
                    param_inputs[key] = st.number_input(label, value=default_values.get(key, 0.0), format="%.4f", key=f"param_{model_id}_{key}")
        else:
            for i, key in enumerate(internal_keys):
                label = all_param_labels[i]
                param_inputs[key] = st.number_input(label, value=default_values.get(key, 1.0), format="%.4f", key=f"param_{model_id}_{key}")

        # Hiển thị tham số được tính toán
        if 'last_calculated_c' in st.session_state and model_id == 'model2':
            st.text_input(tr('model2_calculated_c_label'), value=f"{st.session_state.last_calculated_c:.6g}", disabled=True)
        if 'last_calculated_r' in st.session_state and model_id == 'model3':
            st.text_input(tr('model3_calculated_r_label'), value=f"{st.session_state.last_calculated_r:.8g}", disabled=True)
        if 'last_calculated_alpha' in st.session_state and model_id == 'model4':
            col_alpha, col_beta = st.columns(2)
            col_alpha.text_input(tr('model4_param_alpha'), value=f"{st.session_state.last_calculated_alpha:.6g}", disabled=True)
            col_beta.text_input(tr('model4_param_beta'), value=f"{st.session_state.last_calculated_beta:.6g}", disabled=True)

        selected_component = 'x'
        if model_id == "model5":
            comp_options = {tr('model5_component_x'): 'x', tr('model5_component_y'): 'y'}
            selected_comp_disp = st.radio(tr('model5_select_component'), options=comp_options.keys(), horizontal=True, key=f"comp_{model_id}")
            selected_component = comp_options[selected_comp_disp]

        # 4. Thực thi
        st.subheader(tr('screen2_actions_group'))
        col_btn1, col_btn2 = st.columns(2)
        run_simulation = col_btn1.button(tr('screen2_init_button'), use_container_width=True, type="primary")
        if col_btn2.button(tr('screen2_refresh_button'), use_container_width=True):
            st.session_state.simulation_results = {}
            st.session_state.validated_params = {}
            for key in ['last_calculated_c', 'last_calculated_r', 'last_calculated_alpha', 'last_calculated_beta']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if run_simulation:
        with st.spinner(tr('screen2_info_area_running')):
            is_valid = True
            if not selected_steps_int:
                st.toast(tr('msg_select_step'), icon='⚠️'); is_valid = False
            if 't₀' in param_inputs and 't₁' in param_inputs and param_inputs['t₁'] <= param_inputs['t₀']:
                st.toast(tr('msg_t_end_error'), icon='⚠️'); is_valid = False
            
            if is_valid:
                for key in ['last_calculated_c', 'last_calculated_r', 'last_calculated_alpha', 'last_calculated_beta']:
                    if key in st.session_state: del st.session_state[key]
                
                prep_ok, prep_data = _prepare_simulation_functions(model_data, param_inputs, selected_method_short)
                if prep_ok:
                    ode_func, exact_callable, y0, t_start, t_end = prep_data
                    results_dict = {}
                    for steps in selected_steps_int:
                        res = _perform_single_simulation(model_data, ode_func, exact_callable, y0, t_start, t_end, selected_method_short, steps, h_float, selected_component)
                        if res: results_dict[steps] = res
                    
                    st.session_state.simulation_results = results_dict
                    st.session_state.validated_params = {
                        'params': param_inputs,
                        'method_short': selected_method_short,
                        'method_steps': selected_steps_int[0] if selected_steps_int else None,
                        'h_target': h_float,
                        'model_id': model_id
                    }
                    st.rerun()

    with col_display:
        results = st.session_state.get('simulation_results', {})
        if not results:
            st.info(tr('screen2_info_area_init'))
        else:
            n_steps = len(results)
            colors = plt.cm.jet(np.linspace(0, 1, max(1, n_steps)))
            
            # --- TẠO CÁC ĐỒ THỊ TRONG BỘ NHỚ ---
            @st.cache_data
            def generate_plots(results_data, lang, model_id, selected_method_short):
                # Hàm này được cache để không vẽ lại nếu dữ liệu không đổi
                figs = {}
                # Đồ thị nghiệm
                fig_sol = Figure(); ax_sol = fig_sol.subplots()
                exact_plotted = False; color_idx = 0
                for step, res in sorted(results_data.items()):
                    method_label = f"{selected_method_short[:2].upper()}-{step}"
                    if not exact_plotted and res.get('exact_sol_plot') is not None:
                        ax_sol.plot(res['t_plot'], res['exact_sol_plot'], color='black', ls='--', label=tr('screen2_plot_exact_label'))
                        exact_plotted = True
                    ax_sol.plot(res['t_plot'], res['approx_sol_plot'], color=colors[color_idx], label=method_label)
                    color_idx += 1
                ax_sol.set_title(tr('screen2_plot_solution_title'))
                ax_sol.set_xlabel(tr('screen2_plot_t_axis'))
                ax_sol.set_ylabel(tr('screen2_plot_value_axis'))
                ax_sol.grid(True, linestyle=':'); ax_sol.legend()
                figs['solution'] = fig_sol

                # Đồ thị sai số và bậc hội tụ
                fig_err_ord, (ax_err, ax_ord) = plt.subplots(1, 2, figsize=(12, 4))
                color_idx = 0
                for step, res in sorted(results_data.items()):
                    method_label = f"{selected_method_short[:2].upper()}-{step}"
                    if res.get('n_values_convergence') is not None and len(res['n_values_convergence']) > 0:
                        ax_err.plot(res['n_values_convergence'], res['errors_convergence'], marker='.', color=colors[color_idx], label=method_label)
                    if res.get('log_h_convergence') is not None and len(res['log_h_convergence']) >= 2:
                        slope = res.get('order_slope', 0)
                        fit_label = tr('screen2_plot_order_fit_label_suffix').format(slope)
                        log_h, log_err = res['log_h_convergence'], res['log_error_convergence']
                        ax_ord.plot(log_h, log_err, 'o', color=colors[color_idx], label=f"{method_label}{tr('screen2_plot_order_data_label_suffix')}")
                        ax_ord.plot(log_h, np.polyval(np.polyfit(log_h, log_err, 1), log_h), '-', color=colors[color_idx], label=fit_label)
                    color_idx += 1
                ax_err.set_title(tr('screen2_plot_error_title')); ax_err.set_xlabel(tr('screen2_plot_n_axis'))
                ax_err.set_ylabel(tr('screen2_plot_error_axis')); ax_err.set_yscale('log')
                ax_err.grid(True, which='both', linestyle=':'); ax_err.legend()
                ax_ord.set_title(tr('screen2_plot_order_title')); ax_ord.set_xlabel(tr('screen2_plot_log_h_axis'))
                ax_ord.set_ylabel(tr('screen2_plot_log_error_axis')); ax_ord.grid(True, linestyle=':'); ax_ord.legend()
                fig_err_ord.tight_layout()
                figs['error_order'] = fig_err_ord
                return figs
            
            # Gọi hàm tạo plot và hiển thị
            generated_figures = generate_plots(tuple(sorted(results.items())), st.session_state.lang, model_id, selected_method_short)
            st.pyplot(generated_figures['solution'])
            st.pyplot(generated_figures['error_order'])

            # --- HIỂN THỊ DỮ LIỆU SỐ (ĐÃ HOÀN THIỆN) ---
            with st.expander(tr('screen2_show_data_button')):
                for step, res in sorted(results.items()):
                    method_label = f"Adam-{selected_method_short} {step} {tr('screen2_info_area_show_data_textCont1')}"
                    slope_str = f"{res.get('order_slope', 'N/A'):.4f}" if isinstance(res.get('order_slope'), float) else "N/A"
                    st.markdown(f"#### {method_label}")
                    st.markdown(f"**{tr('screen2_info_area_show_data_order')}** {slope_str}")
                    
                    t = res.get('t_plot')
                    approx = res.get('approx_sol_plot')
                    exact = res.get('exact_sol_plot')
                    
                    if t is not None and approx is not None and len(t) > 0:
                        df_data = {'t': t, tr('screen2_info_area_show_data_approx'): approx}
                        if exact is not None:
                            df_data[tr('screen2_info_area_show_data_exact')] = exact
                            df_data[tr('screen2_info_area_show_data_error')] = np.abs(approx - exact)
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df.head(15).style.format("{:.6f}"), use_container_width=True)
                    else:
                        st.write(tr("screen2_info_area_no_points"))
                    st.markdown("---")
            
            # --- NÚT LƯU HÌNH ẢNH (ĐÃ HOÀN THIỆN) ---
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                # Lưu đồ thị nghiệm
                buf_sol = io.BytesIO()
                generated_figures['solution'].savefig(buf_sol, format="png", dpi=300)
                zf.writestr("solution_plot.png", buf_sol.getvalue())
                
                # Lưu đồ thị sai số/bậc
                buf_err = io.BytesIO()
                generated_figures['error_order'].savefig(buf_err, format="png", dpi=300)
                zf.writestr("error_and_order_plots.png", buf_err.getvalue())

            st.download_button(
                label=tr('screen2_save_button'),
                data=zip_buffer.getvalue(),
                file_name=f"simulation_plots_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
# ==============================================
#           PHẦN 4: TRANG MÔ PHỎNG ĐỘNG
# ==============================================
class Cell:
    def __init__(self, x, y, gen=0):
        self.x = x
        self.y = y
        self.gen = gen
        self.last_division = -100
		
def show_dynamic_simulation_page():
    validated_params = st.session_state.get('validated_params', {})
    if not validated_params:
        st.error("Không có dữ liệu hợp lệ. Vui lòng chạy lại mô phỏng ở trang trước.")
        if st.button(tr('screen3_back_button')):
            st.session_state.page = 'simulation'
            st.rerun()
        return

    model_data = MODELS_DATA[st.session_state.selected_model_key]
    model_id = model_data.get("id", "")
    
    # --- KHỞI TẠO TRẠNG THÁI CHO TRANG NÀY ---
    if 'anim_running' not in st.session_state:
        st.session_state.anim_running = False
    if 'anim_frame' not in st.session_state:
        st.session_state.anim_frame = 0
    # Dùng key riêng cho từng model để reset chính xác
    # Highlight: Thêm key anim_init riêng cho từng model
    if f'anim_init_{model_id}' not in st.session_state:
        st.session_state[f'anim_init_{model_id}'] = True
    
    # --- HEADER ---
    st.title(tr('screen3_dyn_only_title'))
    col_h1, col_h2 = st.columns([1, 1])
    if col_h1.button(f"ᐊ {tr('screen3_back_button')}"):
        st.session_state.page = 'simulation'
        st.session_state.anim_running = False # Dừng animation khi quay lại
        st.rerun()
    if col_h2.button(f"ᐊᐊ {tr('screen3_double_back_button')}"):
        st.session_state.page = 'model_selection'
        st.session_state.simulation_results = {}; st.session_state.validated_params = {}
        st.session_state.anim_running = False # Dừng animation khi quay lại
        st.rerun()
    st.markdown("---")
    
    # --- LAYOUT CHÍNH ---
    col_controls, col_display = st.columns([1, 2])

    with col_controls:
        st.subheader(tr('screen3_settings_group_title'))
        
        # Lựa chọn kịch bản cho Model 5
        if model_id == 'model5':
            if 'm5_scenario' not in st.session_state:
                st.session_state.m5_scenario = 1 
            
            scenario_options = {tr("screen3_sim1_name_m5"): 1, tr("screen3_sim2_name_m5"): 2}
            
            def on_scenario_change():
                # Highlight: Thêm hàm callback để reset animation khi đổi kịch bản
                st.session_state.anim_running = False
                st.session_state.anim_frame = 0
                st.session_state[f'anim_init_{model_id}'] = True

            selected_scenario_disp = st.radio(
                tr("screen3_sim_list_group_title"),
                options=scenario_options.keys(),
                key="m5_scenario_selector",
                on_change=on_scenario_change # Gọi callback khi thay đổi
            )
            st.session_state.m5_scenario = scenario_options[selected_scenario_disp]

        # Điều khiển animation
        speed_multiplier = st.slider(tr('screen3_speed_label'), min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1fx")
        
        c1, c2, c3 = st.columns(3)
        if c1.button('▶️ Play', use_container_width=True):
            st.session_state.anim_running = True
            st.session_state[f'anim_init_{model_id}'] = False
            st.rerun() # Chạy lại để vào vòng lặp while
        if c2.button('⏸️ Pause', use_container_width=True):
            st.session_state.anim_running = False
            st.rerun() # Chạy lại để thoát vòng lặp while
        if c3.button('🔄 Reset', use_container_width=True):
            st.session_state.anim_running = False
            st.session_state.anim_frame = 0
            st.session_state[f'anim_init_{model_id}'] = True
            # Highlight: Xóa các instance cũ khi reset
            if 'abm_instance' in st.session_state: del st.session_state['abm_instance']
            if 'model2_cells' in st.session_state: del st.session_state['model2_cells']
            st.rerun()
        
        st.subheader(tr('screen3_results_group_title'))
        info_placeholder = st.empty()

    with col_display:
        plot_placeholder = st.empty()
    
    results = st.session_state.get('simulation_results', {})
    highest_step = max(results.keys()) if results else None
    sim_data = results[highest_step] if highest_step is not None else {}
    
    # =========================================================================
    # Highlight: BẮT ĐẦU VÒNG LẶP ANIMATION (THAY THẾ TOÀN BỘ VÒNG LẶP CŨ)
    # =========================================================================
    while st.session_state.anim_running:
        current_frame = st.session_state.anim_frame
        
        fig = Figure(figsize=(8, 7)); ax = fig.subplots()
        animation_ended = False

        if model_id == 'model2':
            t_data = sim_data.get('t_plot')
            y_data = sim_data.get('approx_sol_plot')
            
            if st.session_state.get(f'anim_init_{model_id}', True) or 'model2_cells' not in st.session_state:
                st.session_state.model2_cells = [Cell(0, 0, gen=0)]
                st.session_state[f'anim_init_{model_id}'] = False

            if current_frame >= len(t_data):
                animation_ended = True
                current_frame = len(t_data) - 1

            target_n = int(round(y_data[current_frame]))
            cells = st.session_state.model2_cells
            
            # Logic phân chia tế bào (đơn giản hóa từ code gốc)
            if len(cells) < target_n:
                num_to_add = target_n - len(cells)
                for _ in range(num_to_add):
                    parent = random.choice(cells)
                    angle = random.uniform(0, 2 * np.pi)
                    new_x = parent.x + np.cos(angle) * 1.1
                    new_y = parent.y + np.sin(angle) * 1.1
                    cells.append(Cell(new_x, new_y, parent.gen + 1))
            st.session_state.model2_cells = cells

            ax.set_title(tr('screen3_model2_anim_plot_title'))
            all_x = [c.x for c in cells]; all_y = [c.y for c in cells]
            if all_x:
                max_coord = max(max(np.abs(all_x)), max(np.abs(all_y))) + 2
                ax.set_xlim(-max_coord, max_coord); ax.set_ylim(-max_coord, max_coord)
            for cell in cells:
                ax.add_patch(MplCircle((cell.x, cell.y), radius=0.5, color='brown', alpha=0.7))
            ax.set_aspect('equal'); ax.axis('off')
            
            with info_placeholder.container():
                c_val = st.session_state.validated_params['params'].get('c_calculated', 'N/A')
                st.metric(label=tr('screen3_result_c'), value=f"{c_val:.4g}" if isinstance(c_val, float) else c_val)
                st.metric(label=tr('screen3_result_mass'), value=f"{len(cells)}")
                st.metric(label=tr('screen3_result_time'), value=f"{t_data[current_frame]:.2f} s")

        elif model_id == 'model3':
            abm_params = model_data.get("abm_defaults", {})
            if 'abm_instance' not in st.session_state or st.session_state.get(f'anim_init_{model_id}', True):
                r_val = st.session_state.get('last_calculated_r', 0.0001)
                ptrans = np.clip(r_val * abm_params.get("r_to_ptrans_factor", 5000), abm_params.get("ptrans_min", 0.01), abm_params.get("ptrans_max", 0.9))
                total_pop = int(validated_params['params']['n'] + 1)
                st.session_state.abm_instance = DiseaseSimulationABM(
                    total_population=total_pop, initial_infected_count_for_abm=1,
                    room_dimension=abm_params.get('room_dimension', 10.0),
                    contact_radius=abm_params.get('contact_radius', 0.55),
                    transmission_prob=ptrans, agent_speed=abm_params.get('base_agent_speed', 0.05)
                )
                st.session_state[f'anim_init_{model_id}'] = False
            
            abm = st.session_state.abm_instance
            ended_by_logic = abm.step()
            
            ax.set_title(tr('screen3_abm_anim_plot_title'))
            ax.set_xlim(0, abm.room_dimension); ax.set_ylim(0, abm.room_dimension)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            s_coords, i_coords = abm.get_display_coords(abm_params['display_max_total'], abm_params['display_sample_size'])
            ax.scatter(s_coords[:, 0], s_coords[:, 1], c='blue', label=tr('screen3_legend_abm_susceptible'))
            ax.scatter(i_coords[:, 0], i_coords[:, 1], c='red', marker='*', s=80, label=tr('screen3_legend_abm_infected'))
            ax.legend()
            
            with info_placeholder.container():
                stats = abm.get_current_stats()
                st.metric(label=tr('screen3_total_pop'), value=stats['total_population'])
                st.metric(label=tr('screen3_susceptible_pop'), value=stats['susceptible_count'])
                st.metric(label=tr('screen3_infected_pop'), value=stats['infected_count'])
                st.metric(label=tr('screen3_model3_simulation_time_label'), value=f"{stats['time_step']} steps")

            if ended_by_logic or current_frame >= abm_params.get('max_steps', 400):
                animation_ended = True
                st.toast("Mô phỏng ABM kết thúc!")

        elif model_id == 'model5':
            t_data = sim_data.get('t_plot')
            if current_frame >= len(t_data):
                animation_ended = True; current_frame = len(t_data) - 1

            if st.session_state.m5_scenario == 1:
                x_path, y_path = sim_data['approx_sol_plot_all_components']
                ax.set_title(tr('screen3_model5_plot_title_sim1'))
                ax.plot(x_path, y_path, 'b--', alpha=0.5, label=tr('screen3_legend_m5s1_path'))
                ax.plot(x_path[current_frame], y_path[current_frame], 'rP', markersize=12, label=tr('screen3_legend_m5s1_boat'))
                d_val = validated_params['params']['x0']
                ax.axvline(0, color='grey', ls=':'); ax.axvline(d_val, color='grey', ls=':')
                ax.set_xlabel(tr('screen3_model5_plot_xlabel_sim1')); ax.set_ylabel(tr('screen3_model5_plot_ylabel_sim1'))
                ax.grid(True); ax.legend(); ax.set_aspect('equal')
                with info_placeholder.container():
                    st.subheader(tr('screen3_info_m5_sim1_title'))
                    st.metric(label=tr('screen3_m5_boat_speed'), value=f"{validated_params['params']['v']:.2f}")
                    st.metric(label=tr('screen3_m5_water_speed'), value=f"{validated_params['params']['u']:.2f}")
                    st.metric(label=tr('screen3_m5_crossing_time'), value=f"{t_data[current_frame]:.2f} s")
                    if animation_ended:
                        final_pos_str = f"({x_path[-1]:.2f}, {y_path[-1]:.2f})"
                        reaches_target_str = tr('answer_yes') if abs(x_path[-1]) < 0.1 else tr('answer_no')
                        st.metric(label=tr('screen3_m5_boat_reaches_target'), value=reaches_target_str)
                        st.metric(label=tr('screen3_m5_boat_final_pos'), value=final_pos_str)
            
            elif st.session_state.m5_scenario == 2:
                ax.set_title(tr('screen3_model5_plot_title_sim2'))
                ax.text(0.5, 0.5, tr('screen3_model5_not_implemented_msg'), ha='center', va='center')
                animation_ended = True
                with info_placeholder.container():
                    st.subheader(tr('screen3_info_m5_sim2_title'))
                    st.info(tr('screen3_model5_not_implemented_msg'))

        plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        if animation_ended:
            st.session_state.anim_running = False
        else:
            st.session_state.anim_frame += 1
            time.sleep(0.1 / speed_multiplier)
            st.rerun()

    # --- VẼ TRẠNG THÁI TĨNH KHI KHÔNG CHẠY ANIMATION ---
    if not st.session_state.anim_running:
        # Code để vẽ lại frame cuối cùng hoặc frame khởi tạo
        # (Phần này về cơ bản là lặp lại logic vẽ một lần từ vòng lặp while)
        current_frame = st.session_state.anim_frame
        fig = Figure(figsize=(8, 7)); ax = fig.subplots()
        if st.session_state.get(f'anim_init_{model_id}', True):
             ax.text(0.5, 0.5, tr("screen3_waiting_for_data"), ha='center', va='center')
             ax.set_xticks([]); ax.set_yticks([])
             with info_placeholder.container(): st.info(tr("screen3_waiting_for_data"))
        else:
            # Vẽ lại frame cuối cùng
            # Highlight: Thêm logic vẽ lại trạng thái cuối cùng cho tất cả models
            if model_id == 'model2':
                cells = st.session_state.get('model2_cells', [Cell(0,0)])
                ax.set_title(tr('screen3_model2_anim_plot_title'))
                all_x = [c.x for c in cells]; all_y = [c.y for c in cells]
                if all_x:
                    max_coord = max(max(np.abs(all_x)), max(np.abs(all_y))) + 2
                    ax.set_xlim(-max_coord, max_coord); ax.set_ylim(-max_coord, max_coord)
                for cell in cells:
                    ax.add_patch(MplCircle((cell.x, cell.y), radius=0.5, color='brown', alpha=0.7))
                ax.set_aspect('equal'); ax.axis('off')
                with info_placeholder.container():
                    c_val = st.session_state.validated_params['params'].get('c_calculated', 'N/A')
                    st.metric(label=tr('screen3_result_c'), value=f"{c_val:.4g}" if isinstance(c_val, float) else c_val)
                    st.metric(label=tr('screen3_result_mass'), value=len(cells))
            elif model_id == 'model3':
                if 'abm_instance' in st.session_state:
                    abm = st.session_state.abm_instance
                    abm_params = model_data.get("abm_defaults", {})
                    ax.set_title(tr('screen3_abm_anim_plot_title'))
                    ax.set_xlim(0, abm.room_dimension); ax.set_ylim(0, abm.room_dimension)
                    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
                    s_coords, i_coords = abm.get_display_coords(abm_params['display_max_total'], abm_params['display_sample_size'])
                    ax.scatter(s_coords[:, 0], s_coords[:, 1], c='blue', label=tr('screen3_legend_abm_susceptible'))
                    ax.scatter(i_coords[:, 0], i_coords[:, 1], c='red', marker='*', s=80, label=tr('screen3_legend_abm_infected'))
                    ax.legend()
                    with info_placeholder.container():
                        stats = abm.get_current_stats()
                        st.metric(label=tr('screen3_total_pop'), value=stats['total_population'])
                        st.metric(label=tr('screen3_susceptible_pop'), value=stats['susceptible_count'])
                        st.metric(label=tr('screen3_infected_pop'), value=stats['infected_count'])
                        st.metric(label=tr('screen3_model3_simulation_time_label'), value=f"{stats['time_step']} steps")
            elif model_id == 'model5':
                 if st.session_state.m5_scenario == 1:
                    t_data = sim_data.get('t_plot')
                    x_path, y_path = sim_data['approx_sol_plot_all_components']
                    last_frame = min(current_frame, len(t_data)-1)
                    ax.set_title(tr('screen3_model5_plot_title_sim1'))
                    ax.plot(x_path, y_path, 'b--', alpha=0.5, label=tr('screen3_legend_m5s1_path'))
                    ax.plot(x_path[last_frame], y_path[last_frame], 'rP', markersize=12, label=tr('screen3_legend_m5s1_boat'))
                    ax.grid(True); ax.legend(); ax.set_aspect('equal')
                    with info_placeholder.container():
                        st.subheader(tr('screen3_info_m5_sim1_title'))
                        st.metric(label=tr('screen3_m5_boat_speed'), value=f"{validated_params['params']['v']:.2f}")
                        st.metric(label=tr('screen3_m5_water_speed'), value=f"{validated_params['params']['u']:.2f}")
                 elif st.session_state.m5_scenario == 2:
                    ax.set_title(tr('screen3_model5_plot_title_sim2'))
                    ax.text(0.5, 0.5, tr('screen3_model5_not_implemented_msg'), ha='center', va='center')
                    with info_placeholder.container():
                        st.subheader(tr('screen3_info_m5_sim2_title'))
                        st.info(tr('screen3_model5_not_implemented_msg'))

        plot_placeholder.pyplot(fig)
        plt.close(fig)

# =========================================================================
# Highlight: KẾT THÚC CẬP NHẬT VÒNG LẶP ANIMATION
# =========================================================================


# Điểm bắt đầu chạy ứng dụng
if __name__ == "__main__":
    main()
