"""Mide referencias en validación. No consulta métricas de prueba ni entrena la CNN."""
from pathlib import Path
import hashlib, json, platform, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from entrenar import load_manifest, rgb_histogram, evaluate, CLASSES

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datos/grocery_publico'
OUT=ROOT/'resultados'
OUT.mkdir(exist_ok=True)

def main():
    start=time.perf_counter()
    data=load_manifest(DATA/'manifest.csv',DATA)
    counts=pd.crosstab(data.split,data.class_name).reindex(['train','val','test'])
    counts.to_csv(OUT/'distribucion_dataset.csv')
    frames={s:data[data.split.eq(s)] for s in ['train','val']}
    x={s:np.stack([rgb_histogram(p) for p in f.absolute_path]) for s,f in frames.items()}
    y={s:f.label.to_numpy(dtype=int) for s,f in frames.items()}
    models={'clase_mas_frecuente':DummyClassifier(strategy='most_frequent'),
            'histograma_rgb_logistica':make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42))}
    results={}
    for name,model in models.items():
        model.fit(x['train'],y['train'])
        results[name]=evaluate(y['val'],model.predict_proba(x['val']))
    metadata={'ejecutado_utc':datetime.now(timezone.utc).isoformat(),'particion_evaluada':'val',
              'train_rows':len(frames['train']),'val_rows':len(frames['val']),'test_rows_sin_evaluar':int(data.split.eq('test').sum()),
              'classes':CLASSES,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'scikit_learn':sklearn.__version__,'elapsed_seconds':time.perf_counter()-start,
              'sha256_manifest':hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest(),
              'cnn_status':'Preparada sin entrenar; no hay métricas de MobileNetV2',
              'source':'https://github.com/marcusklasson/GroceryStoreDataset'}
    (OUT/'referencias_validacion.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    (OUT/'metadatos_referencias.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(counts.to_string())
    print(json.dumps({k:{m:v[m] for m in ['accuracy','macro_f1']} for k,v in results.items()},indent=2))
    print('Prueba reservada sin evaluar; MobileNetV2 pendiente de Colab.')

if __name__=='__main__':main()
