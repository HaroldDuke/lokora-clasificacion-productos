"""Adapta el ZIP oficial sin mezclar imágenes icónicas con fotografías naturales."""
from pathlib import Path
from zipfile import ZipFile
import argparse, csv, hashlib, json
from collections import Counter

SOURCE='https://github.com/marcusklasson/GroceryStoreDataset'
MAPPING={'Fruit':'frutas','Vegetables':'verduras','Packages':'envasados'}


def prepare(archive: Path, output: Path):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; seen={}; duplicates=[]
    with ZipFile(archive) as z:
        root=next(n[:-len('dataset/classes.csv')] for n in z.namelist() if n.endswith('/dataset/classes.csv'))
        for name in ['README.md','LICENSE']:
            (output/('GROCERY_'+name)).write_bytes(z.read(root+name))
        for name in ['classes.csv','train.txt','val.txt','test.txt']:
            (output/('original_'+name)).write_bytes(z.read(root+'dataset/'+name))
        # Se conserva primero entrenamiento, luego validación y al final prueba.
        # Un duplicado exacto posterior se excluye y se registra, sin consultar el modelo.
        for split in ['train','val','test']:
            for line in z.read(root+'dataset/'+split+'.txt').decode().splitlines():
                if not line.strip():continue
                relative,fine,coarse=[x.strip() for x in line.split(',')]
                parts=Path(relative).parts
                if len(parts)<3 or parts[0]!=split or parts[1] not in MAPPING:
                    raise ValueError('Ruta no esperada en la lista oficial: '+relative)
                raw=z.read(root+'dataset/'+relative)
                digest=hashlib.sha256(raw).hexdigest()
                if digest in seen:
                    duplicates.append({'excluded':relative,'kept':seen[digest],'sha256':digest})
                    continue
                seen[digest]=relative
                target=(output/relative).resolve()
                if not target.is_relative_to(output.resolve()):raise ValueError('Ruta fuera del destino')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
                rows.append({'image_path':relative,'class_name':MAPPING[parts[1]],
                             'product_id':'public_image:'+relative,'source':SOURCE,'license':'MIT (LICENSE del repositorio)',
                             'official_split':split,'group_status':'identity_not_available',
                             'fine_label':fine,'coarse_label':coarse,'sha256':digest})
    manifest=output/'manifest.csv'
    with manifest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'source':SOURCE,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
             'mapping':MAPPING,'images_retained':len(rows),'exact_duplicates_removed':len(duplicates),
             'counts':dict(sorted(Counter(r['official_split']+'/'+r['class_name'] for r in rows).items())),
             'split_protocol':'Listas oficiales train/val/test; exclusión previa de duplicados exactos si existen',
             'group_limit':'Los metadatos no identifican el objeto físico o la sesión; product_id es un identificador de imagen y no acredita independencia por SKU.'}
    (output/'resumen_dataset.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (output/'duplicados_excluidos.json').write_text(json.dumps(duplicates,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args.zip,args.output)
