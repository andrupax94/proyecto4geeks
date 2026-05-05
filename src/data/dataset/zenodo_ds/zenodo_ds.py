import pandas as pd
import os
import json

def generar_csv_audio(df, col_labels, col_mids, col_fname, lista_clases, ruta_carpeta, labels_sinonimous=None, nombre_salida="dataset_procesado.csv"):
    """
    Procesa un dataframe de audio, filtra por clases y añade columnas en singular.
    """
    # 0. Preparar mapa de sinónimos
    mapa_sinonimos = {}
    mapa_mids = {} # Para poder encontrar el MID correspondiente a la clase filtrada
    
    if labels_sinonimous:
        for nombre_oficial, lista_sinonimos in labels_sinonimous.items():
            for sinonimo in lista_sinonimos:
                mapa_sinonimos[sinonimo.strip()] = nombre_oficial

    # --- FUNCIONES DE APOYO ---
    def obtener_labels_limpias(row):
        # Retorna lista de nombres humanos normalizados
        etiquetas_orig = [e.strip() for e in str(row[col_labels]).split(',')]
        return list(set([mapa_sinonimos.get(e, e) for e in etiquetas_orig]))

    def obtener_clase_principal(etiquetas_fila):
        # Encuentra cuál de las clases de 'lista_clases' está presente en la fila
        for clase in lista_clases:
            if clase in etiquetas_fila:
                return clase
        return None

    # 1. Procesar nombres humanos y filtrar
    df['temp_human_list'] = df.apply(obtener_labels_limpias, axis=1)
    
    # Filtrar: Solo filas que tengan alguna clase de la lista
    df_filtrado = df[df['temp_human_list'].apply(lambda x: any(c in x for c in lista_clases))].copy()

    # 2. Crear columnas en SINGULAR (la clase que causó el filtrado)
    df_filtrado['human_label'] = df_filtrado['temp_human_list'].apply(obtener_clase_principal)
    
    # Para el 'label' (MID) en singular, buscamos el MID que corresponde a la posición de la clase
    def obtener_mid_singular(row):
        mids_lista = [m.strip() for m in str(row[col_mids]).split(',')]
        labels_orig = [e.strip() for e in str(row[col_labels]).split(',')]
        clase_detectada = row['human_label']
        
        # Buscamos qué etiqueta original mapeó a la clase detectada
        for i, orig in enumerate(labels_orig):
            if mapa_sinonimos.get(orig, orig) == clase_detectada:
                return mids_lista[i] if i < len(mids_lista) else mids_lista[0]
        return mids_lista[0]

    df_filtrado['label'] = df_filtrado.apply(obtener_mid_singular, axis=1)

    # 3. Formatear columnas de AUDIO y PATH
    df_filtrado['audio'] = df_filtrado[col_fname].apply(lambda x: f"{x}.wav" if not str(x).lower().endswith('.wav') else str(x))
    df_filtrado['path'] = df_filtrado['audio'].apply(lambda x: os.path.join(ruta_carpeta, x))

    # 4. Formatear columnas de LISTAS (Plural)
    df_filtrado['labels'] = df_filtrado[col_mids].apply(lambda x: json.dumps([i.strip() for i in str(x).split(',')]))
    df_filtrado['human_labels'] = df_filtrado['temp_human_list'].apply(json.dumps)

    # 5. Seleccionar y ordenar columnas finales
    columnas_finales = ['audio', 'label', 'labels', 'human_label', 'human_labels', 'path']
    resultado = df_filtrado[columnas_finales]

    # Guardar a CSV en la ruta absoluta especificada
    resultado.to_csv(nombre_salida, index=False, quoting=1)
    
    print(f"Éxito. Archivo guardado en: {os.path.abspath(nombre_salida)}")
    return resultado

# --- EJEMPLO DE CONFIGURACIÓN ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cargar tu dataframe (asegúrate de que dev.csv esté en la misma carpeta)
    try:
        df_original = pd.read_csv(os.path.join(script_dir, "dev.csv"))
     
        sinonimos = {
            "car_horn": ["Vehicle_horn_and_car_horn_and_honking"],
            "gun_shot": ["Gunshot_and_gunfire"],
            "siren":["Siren"],
            "dog_bark":["Dog"],
            "engine":["Idling","Engine"],
            "drilling":["Drill"],
            "alarm":["Alarm"],
            "doorbell":["Doorbell"],
            "telephone":["Telephone"],
            "crying_and_sobbing":["Crying_and_sobbing"],
            "screaming":["Screaming"],
            "shout":["Shout"],
            "traffic_noise":["Traffic_noise_and_roadway_noise"],
        }

        # Clases que quieres conservar (ahora puedes usar los nombres oficiales)
        clases_objetivo = ["car_horn","gun_shot","siren","dog_bark","engine","drilling","alarm","doorbell","telephone","crying_and_sobbing","screaming","shout","traffic_noise"] 
        df_res = generar_csv_audio(
            df=df_original,
            col_labels='labels',
            col_mids='mids',
            col_fname='fname',
            lista_clases=clases_objetivo,
            ruta_carpeta="audios/train",
            labels_sinonimous=sinonimos,
            nombre_salida=os.path.join(script_dir, "dataset_final.csv")
        )
        print(df_res.head())
    except FileNotFoundError:
        print("No se encontró dev.csv en la carpeta del script.")