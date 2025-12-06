# Alitas Factory – Visión artificial para analítica de clientes

Proyecto de la **Maestría en Inteligencia Artificial Aplicada (MIAA) – Universidad Icesi**, que aplica modelos de visión por computador al caso real del restaurante **Alitas Factory** (sur de Cali, Colombia).

El objetivo principal es **contar automáticamente las personas presentes en el restaurante a partir de video**, comparando el conteo automático con un conteo manual (ground truth), como primer paso hacia una solución de analítica de aforo y comportamiento de clientes.

---

## 🎯 Contexto de negocio

- **Problema actual**  
  - Alitas Factory cuenta con cámaras de seguridad, pero **no utiliza analítica de video**.  
  - No se conoce con precisión:
    - Cuántos clientes entran por franja horaria.
    - Cuánto tiempo permanecen.
    - En qué momentos se satura el servicio.

- **Oportunidad**  
  - La **visión artificial** permite detectar y contar personas de forma automática a partir del video.  
  - Con esos datos se pueden generar insights como:
    - Afluencia por día/hora.
    - Nivel de ocupación del local.
    - Soporte para planificación de turnos y promociones.

- **Alcance de este proyecto (MIAA)**  
  - Usar dos familias de modelos SOTA para **detección de personas**:
    - Un modelo tipo **YOLOv11**.
    - Un modelo tipo **Transformer para detección (DETR)**.
  - Procesar un segmento del video (≈ 5 minutos muestreado cada 2 s).
  - Comparar el **conteo automático** vs. el **conteo humano** mediante métricas de error.

Este prototipo se enfoca en el **conteo de personas**, pero sienta las bases para extensiones futuras como:
- seguimiento de individuos (tracking) y tiempo de permanencia,  
- mapas de calor de ocupación por zonas,  
- clasificación demográfica (hombres/mujeres, rangos de edad),  
- integración con datos de ventas (POS).

---

## 🧱 Arquitectura general

A alto nivel, el flujo del proyecto es:

1. **Datos de entrada**  
   - Videos de cámara fija en el interior de Alitas Factory.
   - Segmento seleccionado de ~5 minutos con alto flujo de personas.

2. **Preparación de datos**  
   - Extracción de frames (ej. 1 frame cada 2 segundos).
   - Preprocesamiento básico (redimensionado, formateo de color).

3. **Modelado**  
   - Modelo YOLOv11 pre-entrenado para detectar la clase `person`.
   - Modelo Transformer (DETR) pre-entrenado para detección de objetos.

4. **Conteo y evaluación**  
   - Conteo automático de personas por frame para cada modelo.
   - Conteo manual (ground truth) realizado por los autores.
   - Cálculo de métricas: MAE, RMSE, sesgo, exactitud con tolerancia ±5 personas.
   - Visualización y análisis de resultados.

5. **Conclusiones de negocio**  
   - Interpretación de los resultados desde la perspectiva del restaurante.
   - Recomendaciones sobre cámaras, layout y posibles usos futuros.

---

## 📁 Estructura del repositorio

```text
alitas-factory-miaa/
├── Notebooks/
│   └── notebook_principal.ipynb    # Notebook con todo el pipeline (EDA, modelos, evaluación, gráficos)
├── Models/
│   ├── README.md                   # Descripción de los modelos (pendiente de completar)
│   ├── yolov11_alitas.pt          # (ejemplo) pesos modelo YOLOv11 para el caso Alitas Factory
│   └── detr_alitas.pth            # (ejemplo) pesos modelo DETR para el caso Alitas Factory
├── Evaluación de modelos vf.xlsx   # Archivo Excel con conteos manuales vs modelos y errores por frame
├── Proyecto Vision artificial - Alitas factory.pptx  # Presentación del proyecto
└── README.md                       # Este archivo


---

⚙️ Requisitos e instalación

Se recomienda usar Python 3.10+ y un entorno virtual.

1. Clonar el repositorio
git clone https://github.com/josue-cobaleda/alitas-factory-miaa.git
cd alitas-factory-miaa

2. Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

3. Instalar dependencias
pip install -r requirements.txt


Si no existe requirements.txt, las dependencias mínimas son:

pip install pandas numpy matplotlib openpyxl ultralytics torch torchvision


Ultralytics y torch pueden requerir versiones específicas según CPU/GPU. Revisar documentación oficial en caso de errores de instalación.

🚀 Cómo ejecutar el análisis

Abre el notebook principal en tu entorno favorito:

Jupyter Lab / Notebook,

VS Code (con extensión de Jupyter),

Google Colab.

Archivo principal:

Notebooks/notebook_principal.ipynb


Ejecuta las celdas en orden:

Entendimiento del negocio y de los datos.

Extracción y muestreo de frames.

Aplicación de modelos (YOLOv11, Transformer).

Lectura de Evaluación de modelos vf.xlsx.

Cálculo de métricas y generación de gráficos.

📊 Evaluación de modelos

La evaluación se realiza comparando, para cada frame muestreado:

Conteo manual (ground truth)

Conteo YOLOv11

Conteo Transformer (DETR)

Métricas utilizadas:

MAE (Mean Absolute Error): error absoluto promedio entre conteo automático y manual.

RMSE (Root Mean Square Error): raíz del error cuadrático medio, penaliza más los errores grandes.

Sesgo (Bias): promedio de (predicción – real); indica si el modelo tiende a sobrecontar o subcontar.

Exactitud con tolerancia ±5: porcentaje de frames en los que la diferencia entre conteo automático y manual es ≤ 5 personas.

El notebook genera:

Curvas de conteo en el tiempo para:

Conteo manual

YOLOv11

Transformer

Gráficos de barras comparando métricas de ambos modelos.

Diagramas de dispersión (real vs predicho) con línea diagonal de referencia.

🧩 Limitaciones y posibles mejoras

Limitaciones actuales:

Se trabaja con un solo punto de vista de cámara y un segmento limitado de video (~5 minutos).

El modelo usa pesos pre-entrenados, sin fine-tuning específico al entorno de Alitas Factory.

No se implementó aún:

tracking de personas (IDs únicos),

métricas de tiempo de permanencia,

segmentación por zonas del local.

Líneas de trabajo futuro:

Entrenar o ajustar los modelos con datos del propio restaurante.

Incorporar multi-object tracking para evitar doble conteo y medir tiempo de permanencia por mesa.

Ampliar la analítica a:

mapas de calor,

detección de filas y tiempos de espera,

clasificación demográfica (hombres/mujeres, rangos de edad).

Integrar la salida de los modelos con:

sistema POS,

paneles de BI,

herramientas de planificación de personal.

👥 Autores

Proyecto desarrollado por estudiantes de la Maestría en Inteligencia Artificial Aplicada (MIAA) – Universidad Icesi:

Erik Vergara

Iván Felipe Morán

Josué Cobaleda

