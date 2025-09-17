import streamlit as st
import pandas as pd
import lasio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from difflib import SequenceMatcher
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import base64

# --- INICIALIZAR EL ESTADO DE LA SESIÓN ---
if 'las_loaded' not in st.session_state:
    st.session_state.las_loaded = False
if 'excel_loaded' not in st.session_state:
    st.session_state.excel_loaded = False
if 'df_log' not in st.session_state:
    st.session_state.df_log = None
if 'capas_df' not in st.session_state:
    st.session_state.capas_df = None
if 'nombre_pozo' not in st.session_state:
    st.session_state.nombre_pozo = "Pozo Desconocido"
if 'las_obj' not in st.session_state:
    st.session_state.las_obj = None
if 'capas_resumen' not in st.session_state:
    st.session_state.capas_resumen = pd.DataFrame()


# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide")
st.title('Control de calidad de cemento')

# --- INTEGRACIÓN DEL LOGO ---
logo_path = 'Logo_CGC.png'  # <--- REEMPLAZA ESTO CON EL NOMBRE DE TU ARCHIVO DE IMAGEN
st.sidebar.image(logo_path, use_column_width=True)

st.sidebar.header('Carga de Datos')

# --- PARÁMETROS FIJOS ---
NOMBRES_CURVA_CBL = ['AMP3FT', 'CBLF', 'CBL']
UMBRAL_AMPLITUD_BUENA_MV = 10
UMBRAL_AMPLITUD_MALA_MV = 20
AMPLITUD_LIBRE_MV = 80
TOC_AMPLITUD_UMBRAL = 10
TOC_LONGITUD_MINIMA_M = 10
INTERVALO_LONGITUD_MINIMA_FT = 1.5
RANGO_SELLO_METROS = 3

# --- FUNCIONES DE AYUDA ---
def find_quality_intervals(df_cementado):
    df_cementado['block'] = (df_cementado['Category'] != df_cementado['Category'].shift()).cumsum()
    intervals = df_cementado.groupby(['Category', 'block']).agg(Top=('DEPTH', 'min'), Base=('DEPTH', 'max')).reset_index()
    longitud_minima_m = INTERVALO_LONGITUD_MINIMA_FT * 0.3048
    intervals['Length_m'] = intervals['Base'] - intervals['Top']
    intervals_filtrados = intervals[intervals['Length_m'] >= longitud_minima_m].copy()
    df_para_excel = intervals_filtrados[intervals_filtrados['Category'].isin(['Malo', 'Medio'])].copy()
    df_para_excel = df_para_excel[['Category', 'Top', 'Base', 'Length_m']].round(2).rename(
        columns={'Category': 'Calidad', 'Top': 'Tope (m)', 'Base': 'Base (m)', 'Length_m': 'Longitud (m)'})
    bad_intervals = df_para_excel[df_para_excel['Calidad'] == 'Malo']
    medium_intervals = df_para_excel[df_para_excel['Calidad'] == 'Medio']
    bad_list = [f"De {row['Tope (m)']:.2f} m a {row['Base (m)']:.2f} m" for _, row in bad_intervals.iterrows()]
    medium_list = [f"De {row['Tope (m)']:.2f} m a {row['Base (m)']:.2f} m" for _, row in medium_intervals.iterrows()]
    return df_para_excel, bad_list, medium_list

def detectar_tope_cemento(df, amp_curve, umbral_mv=TOC_AMPLITUD_UMBRAL, longitud_min_m=TOC_LONGITUD_MINIMA_M):
    df_toc = df[[amp_curve]].dropna().sort_index()
    if df_toc.empty:
        st.warning(f"ADVERTENCIA: No hay datos para la curva '{amp_curve}' para detectar TOC. Usando la profundidad mínima del registro.")
        return df.index.min()

    if not pd.api.types.is_numeric_dtype(df_toc.index):
        df_toc.index = pd.to_numeric(df_toc.index, errors='coerce')
        df_toc.dropna(inplace=True)
        if df_toc.empty:
            st.warning(f"ADVERTENCIA: Índice no numérico y no se pudo convertir. Usando la profundidad mínima del registro.")
            return df.index.min()

    df_toc['cumple_umbral'] = df_toc[amp_curve] < umbral_mv
    df_toc['bloque_continuo'] = (df_toc['cumple_umbral'] != df_toc['cumple_umbral'].shift()).cumsum()

    longitudes_bloques = df_toc[df_toc['cumple_umbral']].groupby('bloque_continuo').apply(lambda x: x.index.max() - x.index.min())

    bloques_validos = longitudes_bloques[longitudes_bloques >= longitud_min_m]

    if bloques_validos.empty:
        if len(df_toc) < 2:
            st.warning(f"ADVERTENCIA: Menos de 2 puntos de datos para detectar TOC. Usando la profundidad mínima del registro.")
            return df.index.min()
        df_toc['GRADIENT'] = df_toc[amp_curve].diff().fillna(0)
        toc_depth = df_toc['GRADIENT'].idxmin()
        st.info(f"Tope de Cemento (TOC) detectado por gradiente en: {toc_depth:.2f} m")
    else:
        primer_bloque_valido_id = bloques_validos.index[0]
        toc_depth = df_toc[df_toc['bloque_continuo'] == primer_bloque_valido_id].index.min()
        st.info(f"Tope de Cemento (TOC) detectado por umbral en: {toc_depth:.2f} m")

    return toc_depth

def calcular_adherencia_intervalo(df_cementado_filtrado, step_abs):
    if df_cementado_filtrado.empty:
        return np.nan, 0
    metros_malos = len(df_cementado_filtrado[df_cementado_filtrado['Category'] == 'Malo']) * step_abs
    metros_medios = len(df_cementado_filtrado[df_cementado_filtrado['Category'] == 'Medio']) * step_abs
    longitud_intervalo = (df_cementado_filtrado.index.max() - df_cementado_filtrado.index.min()) + step_abs
    M = metros_malos + metros_medios
    if longitud_intervalo > 0:
        A = 1 - (M / longitud_intervalo)
        A = np.clip(A, 0, 1)
        return A, longitud_intervalo
    else:
        return np.nan, 0

def calcular_kpis_cemento(df_cementado, step, toc_encontrado, capas_pozo_actual, toc_solicitado=None, altura_anillo_solicitado=None):
    step_abs = abs(step)
    metros_buenos = len(df_cementado[df_cementado['Category'] == 'Bueno']) * step_abs
    metros_medios = len(df_cementado[df_cementado['Category'] == 'Medio']) * step_abs
    metros_malos = len(df_cementado[df_cementado['Category'] == 'Malo']) * step_abs
    metros_totales = metros_buenos + metros_medios + metros_malos

    kpis = {
        'TOC_Encontrado': toc_encontrado,
        'Total_Meters': metros_totales,
        'Good_Meters': metros_buenos,
        'Medium_Meters': metros_medios,
        'Bad_Meters': metros_malos,
        'Good_Bond_%': (metros_buenos / metros_totales) * 100 if metros_totales > 0 else 0
    }

    T, A, E, Apnz, Asello = np.nan, np.nan, np.nan, np.nan, np.nan

    if pd.notna(toc_solicitado) and pd.notna(altura_anillo_solicitado) and altura_anillo_solicitado > 0:
        kpis['TOC_Solicitado'] = toc_solicitado
        kpis['Altura_Anillo_Solicitado'] = altura_anillo_solicitado
        kpis['Diferencia_TOC'] = toc_encontrado - toc_solicitado

        diferencia_toc = toc_solicitado - toc_encontrado
        if 0 <= diferencia_toc <= 40.0: T = 1.0
        elif diferencia_toc > 40.0: T = 0.9
        elif -40.0 <= diferencia_toc < 0: T = 0.8
        elif diferencia_toc < -40.0: T = 0.0
        else: T = np.nan
        kpis['TOC_Evaluacion_T'] = T * 100

        M = metros_malos + metros_medios
        if altura_anillo_solicitado > 0:
            A = 1 - (M / altura_anillo_solicitado)
            A = np.clip(A, 0, 1)
            kpis['Adherencia_A'] = A * 100
        else:
            kpis['Adherencia_A'] = np.nan
            A = np.nan
    else:
        kpis['TOC_Solicitado'] = np.nan
        kpis['Altura_Anillo_Solicitado'] = np.nan
        kpis['Diferencia_TOC'] = np.nan
        kpis['TOC_Evaluacion_T'] = np.nan
        kpis['Adherencia_A'] = np.nan

    if pd.notna(T) and pd.notna(A):
        E = (30 * T) + (70 * A)
        E = np.clip(E, 0, 100)
        kpis['Cement_Score'] = E
    else:
        kpis['Cement_Score'] = np.nan

    if not capas_pozo_actual.empty:
        total_M_capas, total_length_capas, total_M_sellos, total_length_sellos = 0, 0, 0, 0
        min_depth_cemented = df_cementado.index.min()
        max_depth_cemented = df_cementado.index.max()

        for _, capa_row in capas_pozo_actual.iterrows():
            capa_top, capa_base = capa_row['Top'], capa_row['Base']
            df_capa_intervalo = df_cementado[(df_cementado.index >= capa_top) & (df_cementado.index <= capa_base)].copy()
            A_capa, longitud_capa = calcular_adherencia_intervalo(df_capa_intervalo, step_abs)

            if pd.notna(A_capa) and longitud_capa > 0:
                total_M_capas += (1 - A_capa) * longitud_capa
                total_length_capas += longitud_capa

            sello_top = max(capa_top - RANGO_SELLO_METROS, min_depth_cemented)
            sello_base = min(capa_base + RANGO_SELLO_METROS, max_depth_cemented)

            if sello_base > sello_top:
                df_sello_intervalo = df_cementado[(df_cementado.index >= sello_top) & (df_cementado.index <= sello_base)].copy()
                A_sello, longitud_sello = calcular_adherencia_intervalo(df_sello_intervalo, step_abs)
                if pd.notna(A_sello) and longitud_sello > 0:
                    total_M_sellos += (1 - A_sello) * longitud_sello
                    total_length_sellos += longitud_sello

        if total_length_capas > 0: Apnz = (1 - (total_M_capas / total_length_capas)) * 100
        else: Apnz = np.nan

        if total_length_sellos > 0: Asello = (1 - (total_M_sellos / total_length_sellos)) * 100
        else: Asello = np.nan

    else:
        Apnz, Asello = np.nan, np.nan

    kpis['Apnz_%'] = np.clip(Apnz, 0, 100) if pd.notna(Apnz) else np.nan
    kpis['Asello_%'] = np.clip(Asello, 0, 100) if pd.notna(Asello) else np.nan

    return pd.Series(kpis)

def get_score_color(score):
    if pd.isna(score): return 'white', 'black'
    if score < 75: return 'red', 'black'
    elif 75 <= score < 85: return '#FFD700', 'black'
    elif 85 <= score < 95: return '#90EE90', 'black'
    else: return '#228B22', 'white'

def normalize_well_name(name):
    if not isinstance(name, str): return ""
    return name.replace(' ', '').replace('_', '').replace('-', '').lower()

# Función para crear el PDF
def create_pdf(kpis_df, df_intervals, df_capas, img_path1, img_path2, well_name):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Verificar si el estilo ya existe antes de agregarlo
    if 'Heading1' not in styles:
        styles.add(ParagraphStyle(name='Heading1', fontSize=18, fontName='Helvetica-Bold'))
    if 'Normal' not in styles:
        styles.add(ParagraphStyle(name='Normal', fontSize=12, fontName='Helvetica'))
    if 'TableHeading' not in styles:
        styles.add(ParagraphStyle(name='TableHeading', fontSize=12, fontName='Helvetica-Bold'))

    story = []

    # Título
    story.append(Paragraph(f"Análisis de Cementación para el pozo: {well_name}", styles['Heading1']))
    story.append(Spacer(1, 0.2 * inch))

    # Resumen de KPIs
    story.append(Paragraph("Resumen de KPIs de Cementación", styles['TableHeading']))
    story.append(Spacer(1, 0.1 * inch))
    kpis_data = [['KPI', 'Valor']] + [[key.replace('_', ' ').title(), f"{value:.2f}"] for key, value in kpis_df.to_dict('records')[0].items()]
    table_kpis = Table(kpis_data)
    table_kpis.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table_kpis)
    story.append(Spacer(1, 0.2 * inch))

    # Gráficos (Ajustado el tamaño)
    story.append(Paragraph("Gráficos del Análisis", styles['TableHeading']))
    story.append(Spacer(1, 0.1 * inch))

    # Aumentado el tamaño de las imágenes en el PDF
    img1 = Image(img_path1, width=7.5*inch, height=3*inch)
    img2 = Image(img_path2, width=7.5*inch, height=8.5*inch)

    story.append(img1)
    story.append(img2)
    story.append(Spacer(1, 0.2 * inch))

    # Tabla de intervalos
    story.append(Paragraph("Intervalos con Calidad Malo y Medio", styles['TableHeading']))
    story.append(Spacer(1, 0.1 * inch))
    df_intervals.columns = ['Ítem', 'Calidad', 'Tope (m)', 'Base (m)', 'Longitud (m)']
    intervals_data = [list(df_intervals.columns)] + df_intervals.values.tolist()
    table_intervals = Table(intervals_data)
    table_intervals.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table_intervals)
    story.append(Spacer(1, 0.2 * inch))

    # Tabla de capas
    if not df_capas.empty:
        story.append(Paragraph("Análisis de Adherencia por Capa", styles['TableHeading']))
        story.append(Spacer(1, 0.1 * inch))

        # Se ajusta la lista de columnas para el PDF
        df_capas.columns = ['Ítem', 'Tope (m)', 'Base (m)', 'Longitud (m)', 'Adherencia (%)', 'Adherencia Sello (%)']

        capas_data = [list(df_capas.columns)] + df_capas.values.tolist()
        table_capas = Table(capas_data)
        table_capas.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table_capas)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# --- CARGA DEL ARCHIVO .LAS ---
uploaded_las_file = st.sidebar.file_uploader("1. Cargar archivo .las", type=['las'])
if uploaded_las_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.las') as tmp_file:
            tmp_file.write(uploaded_las_file.getvalue())
            tmp_las_path = tmp_file.name

        st.session_state.las_obj = lasio.read(tmp_las_path)
        st.session_state.df_log = st.session_state.las_obj.df().sort_index()

        try:
            st.session_state.nombre_pozo = st.session_state.las_obj.well.WELL.value
        except:
            st.session_state.nombre_pozo = "Pozo Desconocido"

        st.session_state.las_loaded = True
        st.success(f"Archivo .las '{uploaded_las_file.name}' cargado exitosamente.")
        os.remove(tmp_las_path)
    except Exception as e:
        st.session_state.las_loaded = False
        st.error(f"Error al leer el archivo .las: {e}")
        if 'tmp_las_path' in locals() and os.path.exists(tmp_las_path):
            os.remove(tmp_las_path)

# --- CARGA DEL ARCHIVO EXCEL (HABILITADO DESPUÉS DE CARGAR EL .LAS) ---
if st.session_state.las_loaded:
    uploaded_excel_file = st.sidebar.file_uploader("2. Cargar Excel de capas", type=['xlsx', 'xls'])
    if uploaded_excel_file:
        try:
            st.session_state.capas_df = pd.read_excel(io.BytesIO(uploaded_excel_file.read()), sheet_name='Capas')
            st.session_state.capas_df.columns = ['Pozo_Excel', 'Top', 'Base']
            st.session_state.capas_df['Top'] = pd.to_numeric(st.session_state.capas_df['Top'], errors='coerce')
            st.session_state.capas_df['Base'] = pd.to_numeric(st.session_state.capas_df['Base'], errors='coerce')
            st.session_state.capas_df.dropna(subset=['Top', 'Base'], inplace=True)
            st.session_state.capas_df = st.session_state.capas_df[st.session_state.capas_df['Base'] > st.session_state.capas_df['Top']]
            st.session_state.excel_loaded = True
            st.success(f"Archivo Excel '{uploaded_excel_file.name}' cargado exitosamente.")
        except Exception as e:
            st.session_state.excel_loaded = False
            st.error(f"Error al leer el archivo Excel: {e}")

# --- ENTRADA DE DATOS ADICIONALES (HABILITADO DESPUÉS DE CARGAR AMBOS ARCHIVOS) ---
if st.session_state.las_loaded and st.session_state.excel_loaded:
    toc_solicitado = st.sidebar.number_input(
        "3. TOC Solicitado (Ts) (m)",
        min_value=0.0,
        format="%.2f",
        help="Ingresar el tope de cemento solicitado en metros (mbbp)"
    )
    altura_anillo_solicitado = st.sidebar.number_input(
        "4. Altura de anillo solicitado (Hs) (mts)",
        min_value=0.0,
        format="%.2f",
        help="Ingresar la altura del anillo solicitado en metros"
    )

    if st.sidebar.button("Analizar Datos"):
        if toc_solicitado and altura_anillo_solicitado:
            st.subheader(f'Analizando pozo: {st.session_state.nombre_pozo}')

            df_log = st.session_state.df_log.copy()
            capas_df = st.session_state.capas_df.copy()

            curva_cbl_encontrada = None
            for nombre_cbl in NOMBRES_CURVA_CBL:
                if nombre_cbl in df_log.columns:
                    curva_cbl_encontrada = nombre_cbl
                    break

            if not curva_cbl_encontrada:
                st.error(f"No se encontró ninguna curva de CBL compatible ({NOMBRES_CURVA_CBL}) en el archivo .las.")
                st.stop()

            df_log['DEPTH'] = df_log.index
            try:
                profundidad_step = abs(st.session_state.las_obj.well.STEP.value) if st.session_state.las_obj.well.STEP.value != 0 else np.median(np.diff(st.session_state.las_obj.df().index))
            except:
                profundidad_step = 0.1524

            toc_depth = detectar_tope_cemento(df_log, curva_cbl_encontrada)
            df_pozo_cementado = df_log[df_log.index >= toc_depth].copy()

            if df_pozo_cementado.empty:
                st.warning("No hay datos en el intervalo cementado. No se pueden generar resultados.")
                st.stop()

            df_pozo_cementado = df_pozo_cementado.interpolate(method='linear', limit_direction='both')
            columnas_necesarias = [curva_cbl_encontrada, 'CLDC', 'NEU']
            columnas_existentes = [col for col in columnas_necesarias if col in df_pozo_cementado.columns]
            df_pozo_cementado.dropna(subset=columnas_existentes, inplace=True)

            if df_pozo_cementado.empty:
                st.warning("No hay datos válidos tras la limpieza. No se pueden generar resultados.")
                st.stop()

            conditions = [
                (df_pozo_cementado[curva_cbl_encontrada] >= UMBRAL_AMPLITUD_MALA_MV),
                (df_pozo_cementado[curva_cbl_encontrada] > UMBRAL_AMPLITUD_BUENA_MV) & (df_pozo_cementado[curva_cbl_encontrada] < UMBRAL_AMPLITUD_MALA_MV)
            ]
            choices = ['Malo', 'Medio']
            df_pozo_cementado['Category'] = np.select(conditions, choices, default='Bueno')

            capas_pozo_actual = capas_df.copy()

            kpis_pozo = calcular_kpis_cemento(df_pozo_cementado, profundidad_step, toc_depth, capas_pozo_actual, toc_solicitado, altura_anillo_solicitado)

            st.subheader('Resumen de KPIs de Cementación')
            kpis_df = kpis_pozo.to_frame(name='Valor').T.round(2)
            st.dataframe(kpis_df)

            # Gráfico de distribución de amplitud
            st.subheader('Distribución de Amplitud de CBL (mV)')
            fig_dist = plt.figure(figsize=(15, 8))
            ax_dist = sns.histplot(df_pozo_cementado, x=curva_cbl_encontrada, hue='Category', palette={'Bueno': 'green', 'Medio': 'yellow', 'Malo': 'red'}, binwidth=1, kde=True, stat="count")
            ax_dist.axvline(x=UMBRAL_AMPLITUD_BUENA_MV, color='black', linestyle='--', linewidth=1.5, label=f'Umbral Bueno ({UMBRAL_AMPLITUD_BUENA_MV} mV)')
            ax_dist.axvline(x=UMBRAL_AMPLITUD_MALA_MV, color='black', linestyle='--', linewidth=1.5, label=f'Umbral Malo ({UMBRAL_AMPLITUD_MALA_MV} mV)')
            ax_dist.set_title('Distribución de CBL con Clasificación por Calidad')
            ax_dist.set_xlabel('Amplitud (mV)')
            ax_dist.set_ylabel('Frecuencia')
            ax_dist.legend(title='Calidad')

            # Agregar métricas al costado del gráfico
            total_metros = kpis_pozo['Total_Meters']
            good_pct = (kpis_pozo['Good_Meters'] / total_metros) * 100 if total_metros > 0 else 0
            medium_pct = (kpis_pozo['Medium_Meters'] / total_metros) * 100 if total_metros > 0 else 0
            bad_pct = (kpis_pozo['Bad_Meters'] / total_metros) * 100 if total_metros > 0 else 0

            text_metrics = (
                f"Total: {total_metros:.2f} m\n\n"
                f"Bueno: {kpis_pozo['Good_Meters']:.2f} m ({good_pct:.1f}%)\n"
                f"Medio: {kpis_pozo['Medium_Meters']:.2f} m ({medium_pct:.1f}%)\n"
                f"Malo: {kpis_pozo['Bad_Meters']:.2f} m ({bad_pct:.1f}%)"
            )

            fig_dist.text(1.02, 0.7, text_metrics, transform=ax_dist.transAxes, fontsize=12,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            st.pyplot(fig_dist)

            st.subheader("Intervalos con Calidad Malo y Medio")
            df_intervals, _, _ = find_quality_intervals(df_pozo_cementado)

            df_intervals_streamlit = df_intervals.copy()
            df_intervals_streamlit.insert(0, 'Ítem', range(1, 1 + len(df_intervals_streamlit)))
            st.dataframe(df_intervals_streamlit.round(2))

            capas_resumen = []
            if not capas_pozo_actual.empty:
                st.subheader('Análisis de Adherencia por Capa')
                for idx, row in capas_pozo_actual.iterrows():
                    capa_top, capa_base = row['Top'], row['Base']

                    # CÁLCULO DE ADHERENCIA DE LA CAPA
                    df_capa_intervalo = df_pozo_cementado[(df_pozo_cementado.index >= capa_top) & (df_pozo_cementado.index <= capa_base)].copy()

                    if not df_capa_intervalo.empty:
                        metros_totales = len(df_capa_intervalo) * profundidad_step
                        metros_malos = len(df_capa_intervalo[df_capa_intervalo['Category'] == 'Malo']) * profundidad_step
                        metros_medios = len(df_capa_intervalo[df_capa_intervalo['Category'] == 'Medio']) * profundidad_step

                        M_capa = metros_malos + metros_medios
                        A_capa = 1 - (M_capa / metros_totales) if metros_totales > 0 else np.nan

                        # CÁLCULO DE ADHERENCIA DEL SELLO
                        sello_top = max(capa_top - RANGO_SELLO_METROS, df_pozo_cementado.index.min())
                        sello_base = min(capa_base + RANGO_SELLO_METROS, df_pozo_cementado.index.max())
                        df_sello_intervalo = df_pozo_cementado[(df_pozo_cementado.index >= sello_top) & (df_pozo_cementado.index <= sello_base)].copy()

                        A_sello, _ = calcular_adherencia_intervalo(df_sello_intervalo, profundidad_step)

                        capas_resumen.append({
                            'Pozo': st.session_state.nombre_pozo,
                            'Capa': f"{row['Pozo_Excel']}",
                            'Tope (m)': capa_top,
                            'Base (m)': capa_base,
                            'Longitud (m)': metros_totales,
                            'Adherencia (%)': np.clip(A_capa, 0, 1) * 100 if pd.notna(A_capa) else np.nan,
                            'Adherencia Sello (%)': np.clip(A_sello, 0, 1) * 100 if pd.notna(A_sello) else np.nan
                        })

                if capas_resumen:
                    df_capas_resumen = pd.DataFrame(capas_resumen).round(2)
                    df_capas_resumen.insert(0, 'Ítem', range(1, 1 + len(df_capas_resumen)))
                    st.dataframe(df_capas_resumen.round(2))

            st.subheader('Reporte de Cementación (Gráfico)')
            # Aumentado el tamaño del gráfico
            fig, axs = plt.subplots(1, 2, figsize=(20, 15), sharey=True, gridspec_kw={'width_ratios': [0.5, 0.5]})

            # Gráfico de Amplitud (izquierda)
            axs[0].plot(df_log[curva_cbl_encontrada], df_log.index, color='purple', lw=0.8)
            axs[0].set_title(f'Amplitud ({curva_cbl_encontrada})', color='black')
            axs[0].set_xlabel('Amplitud (mV)', color='black')
            axs[0].set_xlim(left=0)
            axs[0].axvline(x=UMBRAL_AMPLITUD_BUENA_MV, color='green', linestyle=':', linewidth=1.5, label=f'Umbral Bueno ({UMBRAL_AMPLITUD_BUENA_MV} mV)')
            axs[0].axvline(x=UMBRAL_AMPLITUD_MALA_MV, color='red', linestyle=':', linewidth=1.5, label=f'Umbral Malo ({UMBRAL_AMPLITUD_MALA_MV} mV)')
            axs[0].tick_params(axis='x', colors='black', top=True, bottom=False, labeltop=True, labelbottom=False)
            axs[0].tick_params(axis='y', colors='black')
            axs[0].set_facecolor("white")
            axs[0].spines['left'].set_color('black')
            axs[0].spines['bottom'].set_color('black')
            axs[0].spines['right'].set_color('black')
            axs[0].spines['top'].set_color('black')
            axs[0].xaxis.set_label_position('top')
            axs[0].xaxis.tick_top()

            # Gráfico de Calidad de Cemento (derecha)
            axs[1].set_title('Calidad de Cemento', color='black')
            axs[1].set_xlabel('Clasificación', color='black')
            axs[1].set_xlim(0, 1)
            axs[1].set_xticks([])
            axs[1].fill_betweenx(df_pozo_cementado.index, 0, 1, where=df_pozo_cementado['Category'] == 'Bueno', color='lime', alpha=0.7, label='Bueno')
            axs[1].fill_betweenx(df_pozo_cementado.index, 0, 1, where=df_pozo_cementado['Category'] == 'Medio', color='yellow', alpha=0.7, label='Medio')
            axs[1].fill_betweenx(df_pozo_cementado.index, 0, 1, where=df_pozo_cementado['Category'] == 'Malo', color='red', alpha=0.7, label='Malo')
            axs[1].legend(loc='upper right', fontsize='small')
            axs[1].tick_params(axis='x', colors='black', top=True, bottom=False, labeltop=True, labelbottom=False)
            axs[1].tick_params(axis='y', colors='black')
            axs[1].set_facecolor("white")
            axs[1].spines['left'].set_color('black')
            axs[1].spines['bottom'].set_color('black')
            axs[1].spines['right'].set_color('black')
            axs[1].spines['top'].set_color('black')
            axs[1].xaxis.set_label_position('top')
            axs[1].xaxis.tick_top()

            # Configuración general de los ejes
            for ax_log in [axs[0], axs[1]]:
                ax_log.axhline(y=toc_depth, color='blue', linestyle='--', linewidth=2, label=f'TOC Encontrado @ {toc_depth:.2f} m')
                if pd.notna(toc_solicitado):
                    ax_log.axhline(y=toc_solicitado, color='orange', linestyle='--', linewidth=2, label=f'TOC Solicitado @ {toc_solicitado:.2f} m')
                ax_log.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax_log.set_ylabel("Profundidad (m)", color='black')
                ax_log.legend(loc='upper right', fontsize=8, facecolor='lightgrey', edgecolor='black')

            # Aplicar la solución: definir explícitamente los límites Y invertidos
            ymin = df_log.index.min()
            ymax = df_log.index.max()
            axs[0].set_ylim(ymax, ymin)
            axs[1].set_ylim(ymax, ymin)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            # Botón de descarga PDF
            st.markdown("---")
            st.subheader("Descargar Reporte")

            # Generar los gráficos en buffers de bytes para Reportlab
            buf_dist = io.BytesIO()
            fig_dist.savefig(buf_dist, format='png', bbox_inches='tight')
            buf_dist.seek(0)
            img_dist_path = buf_dist

            buf_report = io.BytesIO()
            fig.savefig(buf_report, format='png', bbox_inches='tight')
            buf_report.seek(0)
            img_report_path = buf_report

            df_intervals_pdf, _, _ = find_quality_intervals(df_pozo_cementado)
            df_intervals_pdf.insert(0, 'Ítem', range(1, 1 + len(df_intervals_pdf)))

            if capas_resumen:
                df_capas_resumen_pdf = pd.DataFrame(capas_resumen).round(2)
                
                # Se eliminan las columnas 'Pozo' y 'Capa' antes de renombrar
                df_capas_resumen_pdf = df_capas_resumen_pdf.drop(columns=['Pozo', 'Capa'])
                
                # Se inserta la columna 'Ítem' al inicio
                df_capas_resumen_pdf.insert(0, 'Ítem', range(1, 1 + len(df_capas_resumen_pdf)))
            else:
                df_capas_resumen_pdf = pd.DataFrame()

            pdf_bytes = create_pdf(kpis_df, df_intervals_pdf, df_capas_resumen_pdf, img_dist_path, img_report_path, st.session_state.nombre_pozo)

            st.download_button(
                label="Descargar Reporte en PDF",
                data=pdf_bytes,
                file_name=f"Reporte_CBL_{st.session_state.nombre_pozo}.pdf",
                mime="application/pdf"
            )

        else:
            st.error("Por favor, ingrese los valores de TOC y Altura de Anillo solicitados.")
else:
    st.info("Por favor, sube un archivo .las y un archivo Excel de capas para comenzar.")