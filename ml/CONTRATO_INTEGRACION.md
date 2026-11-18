# Integración propuesta con Lokora

Este contrato describe una extensión. El ZIP original y los servicios desplegados no se han modificado.

## Consulta de categoría

- Servicio Python propuesto: `POST /predict-category`, multipart con campo `image`.
- Entrada: una fotografía de un producto de envasados, frutas o verduras, máximo 5 MB y 25 megapíxeles.
- Transformación: orientación EXIF, RGB, `resize_with_pad` de TensorFlow a 224 × 224. El modelo incorpora el escalado de [0,255] a [-1,1]. No volver a normalizar desde Express.
- Salida: `suggested_category`, `scores`, `max_softmax`, `model_version`, `inference_ms` y `requires_human_confirmation=true`.
- `scores` son salidas softmax sin calibración demostrada; no interpretarlas como certeza ni como detección de productos fuera del alcance.

## Cambios futuros en la aplicación

1. El vendedor selecciona una foto y solicita una sugerencia para un producto dentro del piloto.
2. Express valida sesión, propiedad de tienda y límites de archivo; reenvía únicamente la imagen al servicio Python, sin credenciales de pago.
3. React muestra la categoría sugerida y permite aceptarla, cambiarla o continuar manualmente. Una indisponibilidad del servicio permite el registro manual.
4. Las clases globales del modelo se mapean a `category_id` de la tienda. No usar el nombre predicho como un ID existente ni crear categorías automáticamente.
5. Solo después de la confirmación se guarda el producto mediante el flujo actual del catálogo.
6. Una extensión de base de datos puede registrar `model_version`, clase sugerida, clase confirmada, tiempos y un identificador de imagen permitido. El registro y la migración no están implementados en la API original.

## Despliegue y MLOps

El servicio carga artefactos locales de un run entrenado y revisado desde `LOKORA_MODEL_DIR`. No descarga pesos durante las solicitudes. Debe mantenerse en una red privada o detrás del backend autenticado antes de habilitar acceso externo. El prototipo no incluye despliegue ni una autorización propia para llamadas públicas.

La promoción del modelo es manual después de revisar métricas, errores y prueba de uso con un vendedor. Guardar la versión anterior para poder revertir. Las correcciones se curan y etiquetan antes de incorporarlas a otro entrenamiento.
