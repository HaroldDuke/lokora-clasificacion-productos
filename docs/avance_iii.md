# Proyecto final: clasificación visual de productos para Lokora

**Avance III (R2-A3-S13). Implementación y pruebas**

Universidad de Cundinamarca. Posgrado. Aplicaciones en Deep Learning (608145)
Docente: Wilson Mauro Rojas Reales
Estudiante: Harold Duque (trabajo individual)
Fecha: 17 de noviembre de 2026

## Resumen

Se entrenó y evaluó MobileNetV2 con aprendizaje por transferencia para sugerir una de tres categorías de producto (envasados, frutas, verduras) a partir de una fotografía. El modelo se comparó con dos referencias: predecir la clase más frecuente y una regresión logística sobre histogramas de color. MobileNetV2 obtuvo un macro-F1 de 0,969 en validación y de 0,969 en prueba, frente a 0,514 y 0,664 de la regresión logística. Los errores se concentran entre frutas y verduras. Estos resultados provienen de un conjunto público cuyos metadatos no permiten garantizar independencia entre particiones ni representan aún fotografías de vendedores de Lokora; por eso se tratan como evidencia preliminar y no como desempeño esperado en producción.

## 1. Implementación

**Datos.** Grocery Store Dataset (Klasson et al., 2019), usando solo fotografías naturales y sus listas oficiales de partición: 2.640 imágenes de entrenamiento, 296 de validación y 2.485 de prueba. Las carpetas Fruit, Vegetables y Packages se mapean a frutas, verduras y envasados.

**Modelo.** MobileNetV2 preentrenada en ImageNet, con la base congelada y una cabeza nueva: pooling global, dropout de 0,2 y capa softmax de tres salidas. Las imágenes se orientan según EXIF, se convierten a RGB, se ajustan con relleno a 224 × 224 y se escalan de 0–255 a -1–1 dentro del modelo. El aumento de datos se aplica solo en entrenamiento.

**Entrenamiento.** Adam con tasa de 0,001, lotes de 32 y un máximo de 20 épocas con parada temprana (paciencia de cuatro épocas sobre la pérdida de validación). La parada temprana no se activó: la pérdida de validación siguió mejorando y se completaron las 20 épocas. El entrenamiento tomó 387 segundos en una GPU Tesla T4 de Google Colab.

| Elemento | Valor |
|---|---|
| Semilla | 42 |
| Python / TensorFlow | 3.13.16 / 2.21.0 |
| scikit-learn / NumPy / pandas | 1.6.1 / 2.4.6 / 2.2.3 |
| Identificador del run | 20261118T011749179143Z |
| Artefactos | modelo.keras, métricas, predicciones, historial, manifiesto con particiones y hashes |

**Trazabilidad.** Cada ejecución guarda el manifiesto con SHA-256, versiones de las bibliotecas, parámetros, historial, predicciones y el hash del artefacto. Los archivos están en la carpeta `resultados/mobilenetv2/` del repositorio (https://github.com/HaroldDuke/lokora-clasificacion-productos).

## 2. Pruebas

**Protocolo.** Los modelos se compararon en validación. La partición de prueba se calculó al final del run, sin ajustar hiperparámetros con ella. La métrica principal es macro-F1, que da el mismo peso a las tres clases; se reportan también exactitud, precisión y recall por clase.

| Modelo | Exactitud (val.) | Macro-F1 (val.) | Exactitud (prueba) | Macro-F1 (prueba) |
|---|---:|---:|---:|---:|
| Clase más frecuente | 0,3750 | 0,1818 | 0,4495 | 0,2067 |
| Histograma RGB y regresión logística | 0,5304 | 0,5136 | 0,6809 | 0,6642 |
| **MobileNetV2 con transferencia** | **0,9696** | **0,9689** | **0,9718** | **0,9694** |

MobileNetV2 supera con claridad a las dos referencias en ambas particiones, por lo que su uso queda justificado frente a ellas. La meta exploratoria de macro-F1 en prueba de 0,80 se cumple, pero era un criterio provisional.

**Resultados por clase de MobileNetV2 en prueba.**

| Clase | Precisión | Recall | F1 | Imágenes |
|---|---:|---:|---:|---:|
| Envasados | 1,000 | 1,000 | 1,000 | 781 |
| Frutas | 0,958 | 0,980 | 0,969 | 1.117 |
| Verduras | 0,961 | 0,918 | 0,939 | 587 |

**Matriz de confusión en prueba** (filas: clase real; columnas: clase predicha).

| Real \ Predicha | Envasados | Frutas | Verduras |
|---|---:|---:|---:|
| Envasados | 781 | 0 | 0 |
| Frutas | 0 | 1.095 | 22 |
| Verduras | 0 | 48 | 539 |

![Figura 1. Matriz de confusión de MobileNetV2](../resultados_mobilenetv2/matriz_confusion.png)

![Figura 2. Curvas de pérdida y exactitud durante el entrenamiento](../resultados_mobilenetv2/curvas_entrenamiento.png)

Las curvas muestran que la pérdida de validación baja de 0,152 a 0,070. Al final, la exactitud de entrenamiento es de 0,995 y la de validación de 0,970. Esa brecha indica un sobreajuste leve que no impide un buen desempeño en prueba (0,972).

## 3. Análisis de errores

De las 2.485 imágenes de prueba, 70 se clasificaron mal (2,8 %): 48 verduras predichas como frutas y 22 frutas predichas como verduras. Ningún envasado se confundió. Los productos con más errores, tomados de la ruta de cada imagen en el conjunto, son:

| Producto | Errores | Imágenes de prueba | Confusión |
|---|---:|---:|---|
| Pimiento (Pepper) | 13 | 110 | verdura → fruta |
| Repollo (Cabbage) | 8 | 19 | verdura → fruta |
| Cebolla (Onion) | 8 | 37 | verdura → fruta |
| Kiwi | 7 | 45 | fruta → verdura |
| Papa (Potato) | 7 | 70 | verdura → fruta |
| Banano (Banana) | 6 | 44 | fruta → verdura |
| Maracuyá (Passion-Fruit) | 4 | 27 | fruta → verdura |

La confianza media (softmax máximo) es de 0,987 en los aciertos y de 0,804 en los errores. Sin embargo, 27 de los 70 errores tienen una confianza de 0,9 o más, así que la puntuación por sí sola no basta para decidir cuándo pedir una revisión adicional al vendedor. Las salidas softmax no tienen calibración demostrada.

## 4. Limitaciones

1. **Independencia entre particiones.** Los metadatos del conjunto no identifican el objeto físico ni la sesión de captura. Fotografías casi idénticas del mismo producto pueden estar repartidas entre entrenamiento y prueba, lo que puede inflar las métricas. No se afirma independencia por producto.
2. **Cambio de dominio.** Las fotografías son de tienda, tomadas con teléfono. No se han probado fotografías reales de vendedores de Lokora.
3. **Alcance.** Solo tres categorías y un producto principal por imagen. El modelo no detecta de forma fiable productos ajenos a ellas.
4. **Convenciones de etiquetado.** Algunos productos son ambiguos (el tomate, por ejemplo, sigue la convención del conjunto y se etiqueta como verdura).
5. **Latencia.** La latencia de inferencia reportada por el run es de 159 ms de mediana y 173 ms en el percentil 95, medida solo sobre un tensor ya preparado. No incluye carga de la imagen, red ni servicio. Debe medirse de nuevo en el despliegue.

## 5. Rediseño propuesto

Las pruebas sugieren estas modificaciones, que se evaluarán en los siguientes avances y no se han implementado todavía:

| Cambio | Motivo | Evidencia que lo motiva |
|---|---|---|
| Agrupar por producto o sesión en fotografías propias | Evitar fugas entre particiones | Limitación 1 |
| Evaluar con fotografías de vendedores de Lokora | Medir el cambio de dominio | Limitación 2 |
| Ajuste fino de las últimas capas de MobileNetV2 | Reducir errores entre frutas y verduras | Sección 3 |
| Probar un modelo basado en Transformers (por ejemplo ViT o CLIP) como segunda alternativa | Comparar con un enfoque distinto | Revisión de literatura (Avance I) |
| Incorporar el nombre del producto como entrada de texto | Resolver casos ambiguos como el pimiento o el tomate | Sección 3 y Zahavy et al. (2016) |
| Umbral de confianza y confirmación del vendedor | Gestionar errores con alta confianza | Sección 3 |

## 6. Siguientes pasos

| Entrega | Contenido |
|---|---|
| Avance IV (23 de noviembre) | Modelo escogido, modelos probados, herramientas y métricas con la interpretación de resultados |
| Avance V y proyecto final (25 de noviembre) | Comparación con el desempeño de una persona, despliegue en la nube con interfaz o API y socialización |

## Referencias

[1] Klasson, M., Zhang, C. y Kjellström, H. (2019). A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels. IEEE Winter Conference on Applications of Computer Vision (WACV). https://arxiv.org/abs/1901.00711

[2] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. y Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR. https://arxiv.org/abs/1801.04381

[3] Zahavy, T., Magnani, A., Krishnan, A. y Mannor, S. (2016). Is a picture worth a thousand words? A Deep Multi-Modal Fusion Architecture for Product Classification in e-commerce. https://arxiv.org/abs/1611.09534

[4] Keras. Transfer learning and fine tuning. https://keras.io/guides/transfer_learning/
