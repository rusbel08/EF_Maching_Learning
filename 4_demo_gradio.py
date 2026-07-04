# ============================================================
# 4_demo_gradio.py — Interfaz profesional mejorada
# Estilo: blanco y verde agrícola
# ============================================================

import json
import numpy as np
import gradio as gr
from tensorflow import keras
from PIL import Image

# ── Configuración ────────────────────────────────────────────
MODEL_PATH = "models/modelo_arandano.keras"
IMG_SIZE   = (224, 224)
UMBRAL_CONFIANZA = 0.60

# ── Cargar modelo y clases ───────────────────────────────────
print("Cargando modelo...")
model = keras.models.load_model(MODEL_PATH)

with open("models/clases.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
nombres_clases = [idx_to_class[i] for i in range(len(idx_to_class))]
print("Clases cargadas:", nombres_clases)

# ── Información por enfermedad ───────────────────────────────
INFO_ENFERMEDAD = {
    "sano": {
        "emoji": "✅",
        "color": "#2E7D32",
        "badge": "SALUDABLE",
        "descripcion": "La planta no muestra signos visibles de enfermedad.",
        "recomendacion": "Continuar con el monitoreo regular del cultivo cada 7-10 días.",
        "sintomas": "Sin manchas, coloración uniforme, frutos o hojas de aspecto normal.",
    },
    "botrytis": {
        "emoji": "⚠️",
        "color": "#E65100",
        "badge": "BOTRYTIS",
        "descripcion": "Moho gris (Botrytis cinerea) — hongo que afecta hojas, flores y frutos.",
        "recomendacion": "Mejorar ventilación, reducir humedad ambiental y aplicar fungicida autorizado.",
        "sintomas": "Manchas pardas con pelusa grisácea, tejido blando y oscurecido.",
    },
    "antracnosis": {
        "emoji": "⚠️",
        "color": "#B71C1C",
        "badge": "ANTRACNOSIS",
        "descripcion": "Antracnosis (Colletotrichum) — hongo que afecta principalmente los frutos.",
        "recomendacion": "Eliminar frutos/hojas afectadas, aplicar control fitosanitario preventivo.",
        "sintomas": "Manchas deprimidas rosadas o anaranjadas, frutos momificados.",
    },
}

# ── CSS profesional agrícola ─────────────────────────────────
CSS = """
/* ── Reset y base ── */
* { box-sizing: border-box; }
footer { display: none !important; }
body, .gradio-container { background: #F1F8F1 !important; font-family: 'Segoe UI', Arial, sans-serif !important; }

/* ── Header principal ── */
.header-box {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(27,94,32,0.25);
    color: white;
}
.header-box h1 { margin: 0 0 6px 0; font-size: 1.7em; font-weight: 700; color: white !important; }
.header-box p  { margin: 0; font-size: 0.95em; color: #C8E6C9; }
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78em;
    color: #E8F5E9;
    margin-top: 8px;
}

/* ── Tarjetas de sección ── */
.card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border: 1px solid #E8F5E9;
    margin-bottom: 14px;
}
.card-title {
    font-size: 0.82em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #388E3C;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #E8F5E9;
}

/* ── Resultado diagnóstico ── */
.result-sano    { border-left: 5px solid #2E7D32 !important; background: #F1F8E9 !important; }
.result-botrytis { border-left: 5px solid #E65100 !important; background: #FFF3E0 !important; }
.result-antracnosis { border-left: 5px solid #B71C1C !important; background: #FFEBEE !important; }
.result-unknown { border-left: 5px solid #757575 !important; background: #FAFAFA !important; }

/* ── Botón principal ── */
button.primary {
    background: linear-gradient(135deg, #2E7D32, #43A047) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    padding: 13px !important;
    box-shadow: 0 3px 10px rgba(46,125,50,0.3) !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 16px rgba(46,125,50,0.4) !important;
}

/* ── Tarjetas de enfermedad ── */
.disease-card {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    border: 1px solid #E0E0E0;
    margin-bottom: 10px;
    transition: box-shadow 0.2s;
}
.disease-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.1); }
.disease-card.sano-card   { border-top: 3px solid #2E7D32; }
.disease-card.bot-card    { border-top: 3px solid #E65100; }
.disease-card.ant-card    { border-top: 3px solid #B71C1C; }
.disease-title { font-weight: 700; font-size: 1em; margin-bottom: 4px; }
.disease-desc  { font-size: 0.88em; color: #555; line-height: 1.5; }

/* ── Métricas del modelo ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
.metric-box {
    flex: 1; min-width: 80px;
    background: #F1F8F1;
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
    border: 1px solid #C8E6C9;
}
.metric-val { font-size: 1.4em; font-weight: 800; color: #2E7D32; }
.metric-lbl { font-size: 0.72em; color: #666; margin-top: 2px; }

/* ── Footer ── */
.footer-box {
    background: #E8F5E9;
    border-radius: 10px;
    padding: 12px 20px;
    text-align: center;
    font-size: 0.82em;
    color: #555;
    margin-top: 10px;
    border: 1px solid #C8E6C9;
}
"""

# ── Función de predicción ────────────────────────────────────
def diagnosticar(imagen):
    if imagen is None:
        html = """<div class='card result-unknown' style='padding:20px'>
        <div style='font-size:2em;margin-bottom:8px'>📷</div>
        <strong>Sube una imagen para comenzar el análisis</strong>
        </div>"""
        return html, None

    img = imagen.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    probs = model.predict(img_array)[0]

    resultado_probs = {nombres_clases[i]: float(probs[i]) for i in range(len(nombres_clases))}
    clase_predicha  = nombres_clases[int(np.argmax(probs))]
    confianza       = float(np.max(probs))

    if confianza < UMBRAL_CONFIANZA:
        html = f"""
        <div class='card result-unknown' style='padding:20px'>
            <div style='font-size:2em;margin-bottom:8px'>❓</div>
            <div style='font-size:1.15em;font-weight:700;color:#424242;margin-bottom:10px'>
                Imagen no reconocida
            </div>
            <div style='font-size:0.9em;color:#666;margin-bottom:12px'>
                Confianza máxima: <strong>{confianza:.1%}</strong>
                (clase más probable: {clase_predicha})
            </div>
            <div style='background:#F5F5F5;border-radius:8px;padding:12px;font-size:0.88em;color:#555'>
                💡 <strong>Sugerencia:</strong> Sube una foto más cercana, con buena iluminación,
                enfocando directamente la hoja o el fruto del cultivo de arándano.
            </div>
        </div>"""
        return html, resultado_probs

    info     = INFO_ENFERMEDAD.get(clase_predicha, {})
    color    = info.get("color", "#388E3C")
    emoji    = info.get("emoji", "")
    badge    = info.get("badge", clase_predicha.upper())
    desc     = info.get("descripcion", "")
    reco     = info.get("recomendacion", "")
    sintomas = info.get("sintomas", "")
    css_cls  = f"result-{clase_predicha}"

    barra_color = color
    barra_ancho = int(confianza * 100)

    html = f"""
    <div class='card {css_cls}' style='padding:22px'>

        <div style='display:flex;align-items:center;gap:14px;margin-bottom:16px'>
            <div style='font-size:2.6em'>{emoji}</div>
            <div>
                <div style='font-size:0.75em;text-transform:uppercase;letter-spacing:0.1em;color:#888;margin-bottom:2px'>Diagnóstico</div>
                <div style='font-size:1.5em;font-weight:800;color:{color}'>{badge}</div>
            </div>
            <div style='margin-left:auto;text-align:right'>
                <div style='font-size:2em;font-weight:900;color:{color}'>{confianza:.1%}</div>
                <div style='font-size:0.72em;color:#888'>Confianza</div>
            </div>
        </div>

        <div style='background:#F9F9F9;border-radius:8px;height:10px;margin-bottom:16px;overflow:hidden'>
            <div style='background:{barra_color};height:100%;width:{barra_ancho}%;border-radius:8px;transition:width 0.6s'></div>
        </div>

        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px'>
            <div style='background:white;border-radius:8px;padding:12px;border:1px solid #EEE'>
                <div style='font-size:0.72em;text-transform:uppercase;color:#888;margin-bottom:4px'>📋 Descripción</div>
                <div style='font-size:0.9em;color:#333'>{desc}</div>
            </div>
            <div style='background:white;border-radius:8px;padding:12px;border:1px solid #EEE'>
                <div style='font-size:0.72em;text-transform:uppercase;color:#888;margin-bottom:4px'>🔬 Síntomas</div>
                <div style='font-size:0.9em;color:#333'>{sintomas}</div>
            </div>
        </div>

        <div style='background:{color}18;border-left:4px solid {color};border-radius:0 8px 8px 0;padding:12px 16px'>
            <div style='font-size:0.75em;text-transform:uppercase;color:{color};font-weight:700;margin-bottom:4px'>
                💊 Recomendación de manejo
            </div>
            <div style='font-size:0.92em;color:#333'>{reco}</div>
        </div>

    </div>"""

    return html, resultado_probs


# ── Interfaz Gradio ───────────────────────────────────────────
with gr.Blocks(title="Detección de Enfermedades en Arándanos", css=CSS, analytics_enabled=False) as demo:

    # Header
    gr.HTML("""
    <div class='header-box'>
        <h1>🫐 Sistema de Detección Temprana de Enfermedades en Arándanos</h1>
        <p>Visión artificial con Transfer Learning (MobileNetV2) para clasificación de enfermedades en cultivos</p>
        <span class='header-badge'>🎓 Trabajo Final — Machine Learning · UPN 2026 · Grupo 3</span>
    </div>
    """)

    with gr.Row():
        # Columna izquierda — entrada
        with gr.Column(scale=1):
            gr.HTML("<div class='card-title'>📷 Imagen del cultivo</div>")
            entrada_imagen = gr.Image(type="pil", label="", height=300)
            boton = gr.Button("🔍 Analizar imagen", variant="primary", size="lg")

            # Métricas del modelo
            gr.HTML("""
            <div class='card' style='margin-top:14px'>
                <div class='card-title'>📊 Rendimiento del modelo</div>
                <div class='metric-row'>
                    <div class='metric-box'>
                        <div class='metric-val'>90.9%</div>
                        <div class='metric-lbl'>Accuracy<br>TEST</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'>95.6%</div>
                        <div class='metric-lbl'>Accuracy<br>Train</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'>96</div>
                        <div class='metric-lbl'>Imágenes<br>Dataset</div>
                    </div>
                    <div class='metric-box'>
                        <div class='metric-val'>3</div>
                        <div class='metric-lbl'>Clases<br>detectadas</div>
                    </div>
                </div>
                <div style='font-size:0.78em;color:#888;text-align:center'>
                    MobileNetV2 · Transfer Learning · TensorFlow 2.16
                </div>
            </div>
            """)

        # Columna derecha — resultado
        with gr.Column(scale=1):
            gr.HTML("<div class='card-title'>🩺 Resultado del diagnóstico</div>")
            salida_html  = gr.HTML()
            salida_probs = gr.Label(label="Probabilidades por clase", num_top_classes=3)

    boton.click(fn=diagnosticar, inputs=entrada_imagen, outputs=[salida_html, salida_probs])
    # Limpiar resultados al eliminar la imagen
    def limpiar():
        return "", None

    entrada_imagen.clear(fn=limpiar, inputs=[], outputs=[salida_html, salida_probs])
    # Sección informativa
    gr.HTML("""
    <div class='card' style='margin-top:6px'>
        <div class='card-title'>📚 Información sobre enfermedades detectadas</div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:14px'>

            <div class='disease-card sano-card'>
                <div class='disease-title' style='color:#2E7D32'>✅ Planta Sana</div>
                <div class='disease-desc'>
                    La planta presenta coloración normal, sin manchas ni deformaciones.
                    Los frutos o hojas tienen apariencia saludable y uniforme.
                    <br><br>
                    <strong>Acción:</strong> Mantener monitoreo preventivo cada 7-10 días.
                </div>
            </div>

            <div class='disease-card bot-card'>
                <div class='disease-title' style='color:#E65100'>⚠️ Botrytis cinerea</div>
                <div class='disease-desc'>
                    Conocida como <em>moho gris</em>, es causada por el hongo
                    <em>Botrytis cinerea</em>. Favorecida por alta humedad y temperaturas frescas.
                    <br><br>
                    <strong>Síntomas:</strong> Manchas pardas con pelusa grisácea.<br>
                    <strong>Acción:</strong> Reducir humedad, mejorar aireación y aplicar fungicida.
                </div>
            </div>

            <div class='disease-card ant-card'>
                <div class='disease-title' style='color:#B71C1C'>⚠️ Antracnosis</div>
                <div class='disease-desc'>
                    Causada por <em>Colletotrichum</em> spp. Afecta principalmente
                    frutos en etapas de maduración bajo condiciones cálidas y húmedas.
                    <br><br>
                    <strong>Síntomas:</strong> Manchas deprimidas rosadas/anaranjadas.<br>
                    <strong>Acción:</strong> Eliminar tejido afectado y aplicar control fitosanitario.
                </div>
            </div>

        </div>
    </div>
    """)

    # Footer
    gr.HTML("""
    <div class='footer-box'>
        🌱 UPN 2026 · Grupo 3 · Proyecto de Machine Learning ·
        Sistema de visión artificial para detección temprana de enfermedades en cultivos de arándanos
    </div>
    """)

# ── Lanzar la app ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(show_api=False)