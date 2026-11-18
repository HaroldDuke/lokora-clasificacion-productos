# Lokora: clasificación de productos a partir de una foto

Cuando alguien publica un producto en una tienda online, casi siempre le toca escoger la categoría a mano. Este proyecto prueba una ayuda sencilla: el vendedor sube la foto, un modelo sugiere la categoría y el vendedor confirma o corrige antes de guardar.

Es la extensión académica de Lokora, la plataforma que hicimos con el equipo Axiom en la hackatón de Open Payments (Medellín, octubre de 2026). Aquí no está la aplicación original; está solo la parte de Deep Learning. Se hizo como proyecto final de Aplicaciones en Deep Learning (Universidad de Cundinamarca, posgrado), de forma individual.

## Dónde está el proyecto hoy

- Hay un modelo entrenado (MobileNetV2 con transferencia de aprendizaje) que separa tres categorías: **envasados, frutas y verduras**.
- Funciona bien sobre un conjunto público de fotos de tienda. Todavía **no** lo he probado con fotos reales de vendedores de Lokora, y ahí es donde más puede fallar.
- El servicio de inferencia (FastAPI) está escrito pero **no está desplegado en la nube ni lo he ejecutado**. Es lo siguiente que falta.

## Resultados

Datos: Grocery Store Dataset, solo fotos naturales, con sus particiones oficiales (2.640 de entrenamiento, 296 de validación, 2.485 de prueba).

| Modelo | Macro-F1 validación | Macro-F1 prueba | Exactitud prueba |
|---|---:|---:|---:|
| Siempre la clase más frecuente | 0,182 | 0,207 | 0,449 |
| Histograma de color + regresión logística | 0,514 | 0,664 | 0,681 |
| MobileNetV2 (transferencia, base congelada) | 0,969 | 0,969 | 0,972 |

De las 2.485 fotos de prueba, 70 salieron mal: 48 verduras que el modelo llamó fruta y 22 frutas que llamó verdura. Con los envasados no se equivocó ni una vez. Los productos que más se confunden son pimiento, repollo, cebolla, kiwi, papa y banano.

Tres advertencias que conviene tener presentes al leer esos números:

1. El conjunto público no dice qué objeto físico aparece en cada foto, así que fotos casi iguales del mismo producto pueden haber quedado repartidas entre entrenamiento y prueba. El 0,97 puede estar algo inflado.
2. Son fotos de tienda tomadas con teléfono, no de los vendedores de Lokora.
3. Hay 27 errores en los que el modelo estaba muy seguro (softmax de 0,9 o más), así que esa puntuación no sirve por sí sola para decidir cuándo pedir una revisión.

El análisis completo está en `docs/avance_iii.pdf`.

## Cómo está organizado

```
ml/               código: preparar datos, referencias, entrenamiento y servicio
notebooks/        notebook para Google Colab
config/           categorías del piloto
datos/            plantillas para fotos propias y para medir tiempos
resultados/       lo que salió de cada ejecución (métricas, gráficas, predicciones, modelo)
diagramas/        arquitectura y pipeline (PNG y fuente Mermaid)
docs/             los tres informes en Markdown, LaTeX y PDF
```

## Cómo reproducirlo

### En Google Colab (lo más fácil)

1. Abre el notebook: [`notebooks/lokora_clasificacion_productos_Harold_Duque.ipynb`](https://colab.research.google.com/github/HaroldDuke/lokora-clasificacion-productos/blob/main/notebooks/lokora_clasificacion_productos_Harold_Duque.ipynb).
2. Entorno de ejecución → Cambiar tipo de entorno → GPU (T4).
3. En la primera celda de código pon `ENTRENAR_CNN=True`.
4. Entorno de ejecución → Ejecutar todo. Descarga unos 118 MB de datos públicos y comprueba su hash. Mi entrenamiento tardó unos 6,5 minutos en una T4.

### En tu computador

Necesitas Python 3 y TensorFlow 2.16 o superior. Desde la raíz del repositorio:

```bash
pip install -r ml/requirements_colab.txt

# 1. Descargar y preparar el conjunto (necesitas el ZIP oficial del Grocery Store Dataset)
python ml/preparar_grocery.py --zip /ruta/GroceryStoreDataset-master.zip --output datos/grocery_publico

# 2. Referencias (clase frecuente y regresión logística), solo con validación
python ml/evaluar_referencias.py

# 3. Entrenar MobileNetV2
python ml/entrenar.py --manifest datos/grocery_publico/manifest.csv --images datos/grocery_publico --output resultados_cnn --epochs 20
```

Cada ejecución guarda sus métricas, las predicciones, el historial, las versiones de las bibliotecas y el hash del modelo, para poder saber después con qué datos y con qué código salió cada resultado.

## Usar el modelo

El servicio recibe una foto y devuelve la categoría sugerida con sus puntuaciones. Esto es lo que debería funcionar, pero **aún no lo he ejecutado**:

```bash
pip install -r ml/requirements_servicio.txt
cd ml
LOKORA_MODEL_DIR=../resultados/mobilenetv2 uvicorn servicio_inferencia:app --port 8000

# en otra terminal
curl -F "image=@mi_foto.jpg" http://localhost:8000/predict-category
```

La respuesta incluye `suggested_category`, `scores`, `max_softmax`, `model_version`, `inference_ms` y `requires_human_confirmation`. La imagen puede pesar hasta 5 MB. El detalle de cómo encajaría en la app de Lokora está en [`ml/CONTRATO_INTEGRACION.md`](ml/CONTRATO_INTEGRACION.md).

## Fotos propias

Para medir cómo le va con fotos de comercios reales, usa la plantilla `datos/manifest_plantilla.csv` (columnas `image_path, class_name, product_id, source, license`). Agrupa las fotos por producto o por sesión de captura, para que un mismo producto no aparezca a la vez en entrenamiento y en prueba. `datos/medicion_tiempos_plantilla.csv` sirve para anotar cuánto tarda alguien con y sin la sugerencia.

## Dependencias

Python 3, TensorFlow/Keras, scikit-learn, pandas, NumPy, Pillow y matplotlib para entrenar. Para el servicio, además FastAPI, Uvicorn y python-multipart. Los rangos de versión están en `ml/requirements_colab.txt` y `ml/requirements_servicio.txt`; las versiones exactas de cada ejecución quedan en `resultados/mobilenetv2/metadatos.json`.

## Los informes

| Entrega | Contenido | Archivos |
|---|---|---|
| Avance I | Problema, cinco papers comparados y diseño | `docs/avance_i.pdf` |
| Avance II | Pipeline, datos y preprocesamiento | `docs/avance_ii.pdf` |
| Avance III | Implementación, pruebas, errores y rediseño | `docs/avance_iii.pdf` |

Cada uno tiene su fuente en Markdown y en LaTeX en la misma carpeta. `docs/md_a_latex.py` convierte los `.md` a `.tex`, y los PDF salen con `tectonic avance_i.tex`.

## Créditos

- **Lokora**, proyecto del equipo Axiom en la hackatón de Open Payments: https://github.com/cataia-code/interledger-axiom-hackathon. Este repositorio no modifica ni incluye ese código.
- **Grocery Store Dataset**: Klasson, Zhang y Kjellström, WACV 2019. Licencia MIT. https://github.com/marcusklasson/GroceryStoreDataset
- **MobileNetV2**: Sandler et al., CVPR 2018. https://arxiv.org/abs/1801.04381

Harold Duque, 2026.
