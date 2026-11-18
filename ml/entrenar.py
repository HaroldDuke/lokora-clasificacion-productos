"""Experimento de clasificación de imágenes para Lokora.

No contiene datos ni métricas prefabricadas. Se ejecuta con un manifiesto real:
python entrenar.py --manifest datos/manifest.csv --images datos --output resultados
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ['envasados', 'frutas', 'verduras']
SEED = 42
SIZE = 224


def load_manifest(manifest: Path, image_root: Path) -> pd.DataFrame:
    data = pd.read_csv(manifest, dtype=str).fillna('')
    required = ['image_path', 'class_name', 'product_id', 'source', 'license']
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    if data.empty or (data[required].apply(lambda s: s.str.strip()).eq('').any().any()):
        raise ValueError('El manifiesto debe contener imágenes reales y completar todos los campos obligatorios.')
    if set(data.class_name) != set(CLASSES):
        raise ValueError(f'El primer experimento requiere exactamente estas clases: {CLASSES}')
    if data.groupby('product_id').class_name.nunique().gt(1).any():
        raise ValueError('Un product_id no puede tener etiquetas de clases diferentes.')
    hashes, paths = [], []
    root = image_root.resolve()
    for rel in data.image_path:
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError(f'Ruta inexistente o fuera de la carpeta de imágenes: {rel}')
        with Image.open(p) as im:
            im.verify()
        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
        paths.append(str(p))
    data['sha256'] = hashes
    data['absolute_path'] = paths
    if data.sha256.duplicated().any():
        raise ValueError('Hay imágenes idénticas. Depurarlas antes de separar los datos.')
    data['label'] = data.class_name.map({c: i for i, c in enumerate(CLASSES)})
    if 'official_split' in data.columns and data.official_split.ne('').all():
        if set(data.official_split) != {'train', 'val', 'test'}:
            raise ValueError('official_split debe contener train, val y test.')
        data['split'] = data.official_split
        for split in ['train', 'val', 'test']:
            if set(data.loc[data.split.eq(split), 'class_name']) != set(CLASSES):
                raise ValueError('Cada partición oficial debe contener las tres clases.')
    else:
        groups = data[['product_id', 'class_name']].drop_duplicates()
        if groups.groupby('class_name').size().reindex(CLASSES, fill_value=0).min() < 15:
            raise ValueError('Se requieren al menos 15 grupos independientes por clase para la partición inicial; se recomiendan 30 o más.')
        trainval, test = train_test_split(groups, test_size=.20, random_state=SEED, stratify=groups.class_name)
        train, val = train_test_split(trainval, test_size=.25, random_state=SEED, stratify=trainval.class_name)
        split_map = {g: s for s, frame in [('train', train), ('val', val), ('test', test)] for g in frame.product_id}
        data['split'] = data.product_id.map(split_map)
    return data


def rgb_histogram(path: str) -> np.ndarray:
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((224, 224)), dtype=np.float32) / 255.
    h = [np.histogram(rgb[:, :, c], bins=16, range=(0, 1), density=False)[0] / (224*224) for c in range(3)]
    return np.concatenate(h + [rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))])


def read_rgb(path):
    if isinstance(path, np.ndarray):
        path = path.item()
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.float32)


def evaluate(y, probabilities):
    pred = np.asarray(probabilities).argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(len(CLASSES)), average='macro', zero_division=0)),
            'classification_report': classification_report(y, pred, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()}


def run(manifest: Path, image_root: Path, output: Path, epochs: int = 20):
    import tensorflow as tf
    import sklearn
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    started = time.perf_counter()
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = output / stamp
    out.mkdir(parents=True, exist_ok=False)
    data = load_manifest(manifest, image_root)
    data.drop(columns='absolute_path').to_csv(out/'manifest_con_particiones.csv', index=False)
    frames = {s: data[data.split.eq(s)].copy() for s in ['train', 'val', 'test']}
    # Las proporciones son por producto; el número de fotografías puede diferir.
    counts=pd.crosstab(data.split, data.class_name).reindex(['train', 'val', 'test'])
    counts.to_csv(out/'distribucion_datos.csv')
    print(counts)
    hist = {s: np.stack([rgb_histogram(p) for p in f.absolute_path]) for s, f in frames.items()}
    labels = {s: f.label.to_numpy(dtype=int) for s, f in frames.items()}
    majority = DummyClassifier(strategy='most_frequent').fit(hist['train'], labels['train'])
    rgb_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=SEED))
    rgb_model.fit(hist['train'], labels['train'])
    joblib.dump(rgb_model, out/'referencia_rgb.joblib')

    def dataset(frame, training=False):
        ds = tf.data.Dataset.from_tensor_slices((frame.absolute_path.to_numpy(), frame.label.to_numpy(dtype=np.int32)))
        def decode(p, y):
            image = tf.numpy_function(read_rgb, [p], tf.float32)
            image.set_shape([None, None, 3])
            return tf.image.resize_with_pad(image, SIZE, SIZE), y
        ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(len(frame), seed=SEED)
        return ds.batch(32).prefetch(tf.data.AUTOTUNE)

    ds = {s: dataset(f, training=(s == 'train')) for s, f in frames.items()}
    base = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = tf.keras.Input(shape=(SIZE, SIZE, 3), name='rgb_0_255')
    x = tf.keras.layers.RandomFlip('horizontal')(inputs)
    x = tf.keras.layers.RandomRotation(.05)(x)
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs, name='lokora_mobilenetv2')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = out/'modelo.keras'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_loss', save_best_only=True)]
    history = model.fit(ds['train'], validation_data=ds['val'], epochs=epochs, callbacks=callbacks)
    model = tf.keras.models.load_model(checkpoint)
    # Comparación y umbral exploratorio se documentan en validación. La prueba
    # se consulta una vez al final, sin reajustar parámetros a partir de ella.
    result = {}
    for split in ['val', 'test']:
        probs = model.predict(ds[split], verbose=0)
        result[split] = {
            'clase_mas_frecuente': evaluate(labels[split], majority.predict_proba(hist[split])),
            'histograma_rgb_logistica': evaluate(labels[split], rgb_model.predict_proba(hist[split])),
            'mobilenetv2': evaluate(labels[split], probs),
        }
        predictions = frames[split][['image_path', 'product_id', 'class_name']].copy()
        predictions['predicted_class'] = np.array(CLASSES)[probs.argmax(axis=1)]
        predictions['max_softmax'] = probs.max(axis=1)
        predictions.to_csv(out/f'predicciones_{split}.csv', index=False)
    best_ref = max(result['val'][k]['macro_f1'] for k in ['clase_mas_frecuente', 'histograma_rgb_logistica'])
    candidate_selected = result['val']['mobilenetv2']['macro_f1'] >= best_ref
    # Metas propuestas para el piloto; no equivalen a desempeño ya alcanzado.
    acceptance = {'seleccionado_en_validacion': bool(candidate_selected),
                  'meta_macro_f1_prueba': .80,
                  'macro_f1_prueba_cumple_meta': result['test']['mobilenetv2']['macro_f1'] >= .80,
                  'estado': 'candidato experimental; requiere revisión de errores y evaluación con vendedor'}
    sample = next(iter(ds['test']))[0][:1]
    model(sample, training=False).numpy()  # calentamiento antes de medir
    latencies=[]
    for _ in range(30):
        t=time.perf_counter(); model(sample, training=False).numpy(); latencies.append((time.perf_counter()-t)*1000)
    metadata = {'run_id': stamp, 'seed': SEED, 'classes': CLASSES, 'input_size': [224,224,3],
                'input_range': [0,255], 'resize': 'tf.image.resize_with_pad',
                'normalization': 'Rescaling(1/127.5, offset=-1) incluida en el modelo',
                'split': ('Listas oficiales del conjunto público; identidad física no disponible' if 'official_split' in data.columns else '60/20/20 de grupos product_id estratificados por clase'),
                'sha256_manifest': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'python': platform.python_version(), 'tensorflow': tf.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn':sklearn.__version__,
                'epochs_run':len(history.history['loss']), 'backbone_trainable':False,
                'device': [d.name for d in tf.config.list_physical_devices()],
                'latency_inference_only_ms_median': float(np.median(latencies)),
                'latency_inference_only_ms_p95': float(np.percentile(latencies,95)),
                'total_seconds': time.perf_counter()-started,
                'artifact_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    for name, content in [('metricas.json',result),('metadatos.json',metadata),('criterios_piloto.json',acceptance),('clases.json',CLASSES)]:
        (out/name).write_text(json.dumps(content,indent=2,ensure_ascii=False), encoding='utf-8')
    pd.DataFrame(history.history).to_csv(out/'historial_entrenamiento.csv',index=False)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for key, title, ax in [('loss','Pérdida',axes[0]),('accuracy','Exactitud',axes[1])]:
        ax.plot(history.history[key],label='Entrenamiento');ax.plot(history.history['val_'+key],label='Validación')
        ax.set_title(title);ax.set_xlabel('Época desde cero');ax.legend()
    fig.tight_layout();fig.savefig(out/'curvas_entrenamiento.png',dpi=180);plt.close(fig)
    cm = np.array(result['test']['mobilenetv2']['confusion_matrix'])
    fig, ax=plt.subplots(figsize=(6,5));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set_xticks(range(3),CLASSES,rotation=20);ax.set_yticks(range(3),CLASSES)
    ax.set_xlabel('Predicción');ax.set_ylabel('Clase real');ax.set_title('Matriz de confusión en prueba')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.tight_layout();fig.savefig(out/'matriz_confusion.png',dpi=180);plt.close(fig)
    print(json.dumps({'run':str(out),'test_macro_f1':result['test']['mobilenetv2']['macro_f1'],'criterios':acceptance},indent=2,ensure_ascii=False))
    return out


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('resultados'))
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    run(args.manifest,args.images,args.output,args.epochs)
