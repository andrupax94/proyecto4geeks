# MIVIA - Sistema de Detección de Sonidos (Despliegue)

Este directorio contiene la solución completa de despliegue para el proyecto **MIVIA** (Monitoreo Inteligente de Vigilancia con Inteligencia Artificial). Hemos complementado el backend con los modelos de Machine Learning solicitados, integrado el preprocesamiento de audio del pipeline híbrido y conectado el frontend (Next.js) con el backend (FastAPI).

---

## 🚀 Arquitectura del Sistema

El sistema implementa un **Pipeline de Inferencia Híbrida de Dos Etapas** basado en la arquitectura **HybridCNN v3** (espectrogramas Mel + Waveform crudo):

1. **Etapa 1 (Clasificación Binaria):** Utiliza el modelo `best_alertable_v3.pt` para determinar si el audio de entrada es **Alertable** o **No Alertable** (con un umbral óptimo de confianza).
2. **Etapa 2 (Clasificación Multiclase):** 
   - Si el audio es **Alertable**, se redirige al modelo `best_human_label_v6.pt` (clasifica en 10 clases críticas como disparos, sirenas, peleas, accidentes, etc.).
   - Si el audio es **No Alertable**, se redirige al modelo `best_no_alertable_v4.pt` (clasifica en 11 clases de ruido cotidiano o ambiental).

---

## 📁 Estructura del Proyecto de Despliegue

```text
deployment/
├── backend/
│   ├── app/
│   │   ├── main.py                   # Punto de entrada de FastAPI y CORS
│   │   ├── models/
│   │   │   └── model_loader.py       # Carga dinámica de modelos PyTorch (.pt) y mapeos (.pkl)
│   │   ├── routes/
│   │   │   ├── prediction_routes.py  # Endpoint de predicción (/predict)
│   │   │   └── dashboard_routes.py   # Endpoints de estadísticas y EDA del dataset
│   │   └── services/
│   │       └── prediction_service.py # Pipeline de inferencia híbrida y preprocesamiento de audio
│   └── requirements.txt              # Dependencias del backend (FastAPI, PyTorch, Torchaudio, etc.)
└── frontend/
    ├── app/
    │   ├── page.tsx                  # Dashboard interactivo con subida de audios y visualización
    │   ├── layout.tsx                # Layout principal de Next.js
    │   └── globals.css               # Estilos globales con Tailwind CSS
    ├── components/
    │   ├── Sidebar.tsx               # Navegación lateral
    │   ├── DashboardCard.tsx         # Tarjetas de estadísticas generales
    │   ├── EDAChart.tsx              # Gráficos de barras interactivos (Recharts)
    │   ├── AudioTable.tsx            # Historial de audios predichos
    │   ├── ModelMetrics.tsx          # Métricas de evaluación detalladas del modelo
    │   ├── WikiSection.tsx           # Wiki informativa del proyecto
    │   └── MobileView.tsx            # Simulación interactiva de vista móvil
    ├── services/
    │   └── api.ts                    # Cliente API para conectar con el backend
    ├── types/
    │   └── index.ts                  # Definiciones de tipos TypeScript
    ├── package.json                  # Dependencias del frontend (Next.js, Recharts, Tailwind)
    └── next.config.ts                # Configuración de Next.js
```

---

## 🛠️ Instrucciones de Ejecución Local

### Paso 1: Configurar el Backend (FastAPI)

1. Abre una terminal y navega al directorio del backend:
   ```bash
   cd deployment/backend
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Inicia el servidor de desarrollo:
   ```bash
   PYTHONPATH=../../ uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *Nota: El backend estará disponible en `http://127.0.0.1:8000`.*

### Paso 2: Configurar el Frontend (Next.js)

1. Abre otra terminal y navega al directorio del frontend:
   ```bash
   cd deployment/frontend
   ```

2. Instala las dependencias de Node.js:
   ```bash
   npm install
   ```

3. Configura la variable de entorno para conectar con el backend. Crea un archivo `.env.local` en la raíz de `frontend/` con:
   ```env
   NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
   ```

4. Inicia el servidor de desarrollo de Next.js:
   ```bash
   npm run dev
   ```
   *Nota: Abre tu navegador en `http://localhost:3000` para interactuar con la aplicación.*

---

## 📊 Endpoints de la API del Backend

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado del backend y lista de endpoints disponibles. |
| `POST` | `/predict` | Recibe un archivo de audio (multipart/form-data) y devuelve la predicción híbrida de dos etapas con el top-K de confianza. |
| `GET` | `/dashboard/stats` | Estadísticas reales del dataset de entrenamiento (129,440 audios, 23 clases, accuracy del 94.1%). |
| `GET` | `/dashboard/eda` | Distribución exacta de las clases alertables y no alertables del dataset final. |

---

## 🧠 Características Destacadas Implementadas

- **Preprocesamiento Robusto:** Integración directa con el módulo `Preprocess` de tu repositorio, aplicando remuestreo automático a 16,000 Hz, normalización de pico a 0.99, y extracción de espectrogramas Mel y coeficientes MFCC tal como lo definiste en tus notebooks de entrenamiento.
- **Inferencia Híbrida:** Lógica inteligente que detecta dinámicamente si el modelo cargado espera `mel_only`, `mel_mfcc` o `mel_waveform` analizando las llaves de su `state_dict`, lo que asegura compatibilidad total con tus modelos de la arquitectura `ImprovedMFCCCNN`.
- **Dashboard Moderno y Completo:** El frontend no es solo una página de carga; incluye un dashboard completo con estadísticas del dataset real, análisis exploratorio interactivo (EDA), métricas detalladas del modelo (con gráfico de radar) y una simulación en tiempo real de cómo se vería la predicción en una aplicación móvil.
