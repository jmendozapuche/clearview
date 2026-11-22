# ClearView

Prototipo para **Meta Quest 3 / 3S** que detecta publicidad en el mundo real (vallas, carteles,
logos, pantallas publicitarias) y la **hace desaparecer** con un parche del color de su entorno, sin
tapar nunca lo que importa para tu seguridad.

> ⚠️ Prototipo de investigación. Úsalo en entornos controlados, no por la calle con tráfico.
> Lee [docs/SAFETY.md](docs/SAFETY.md).

## Cómo funciona

```
Cámara passthrough ──► YOLO (en el visor) ──► tracker ──► política de seguridad ──► parches anclados al mundo
 (Passthrough Camera API)  anuncios + personas,     estabiliza    descarta lo dudoso      color del entorno,
                           vehículos, señales       detecciones                           bordes suaves
```

1. **Captura:** se toma un frame de la cámara junto con la pose y la calibración de la cámara en ese instante.
2. **Detección:** un YOLO11n corre en la GPU del Quest (Unity Inference Engine), unas pocas capas por
   frame para no bajar los FPS. Detecta dos tipos de cosas: **anuncios** (candidatos a ocultar) y
   **objetos de seguridad** (que nunca se tapan).
3. **Tracking:** un anuncio tiene que verse varias veces seguidas antes de ocultarse.
4. **Política de seguridad:** 14 reglas (no tapar personas, vehículos ni señales; nada cercano; nada
   a ras de suelo; nada enorme; desactivar al girar la cabeza rápido...). Ver [SAFETY.md](docs/SAFETY.md).
5. **Colocación:** se lanza un rayo desde la cámara contra la profundidad del entorno (Depth API / MRUK)
   para encontrar la pared real. El parche se coloca sobre ella, incluso si la ves en ángulo.
6. **Relleno:** el parche mezcla los colores que hay justo fuera de cada borde del anuncio y se funde
   con bordes difuminados. Así parece pared vacía y no un bloque.

## Estructura

```
unity/Assets/ClearView/
  Core/      Lógica pura en C#, sin Unity: decodificador YOLO, tracker, reglas de seguridad,
             guardia de movimiento, watchdog, muestreo de color. 100 % testeada.
  Runtime/   Componentes de Unity: cámara, detector, proyección al mundo, render, controlador.
  Shaders/   ClearView/SoftFill: relleno de 4 colores con bordes suaves, compatible con stereo instanced.
tests/       Tests xUnit del Core (dotnet test).
ml/          Pipeline de datos, entrenamiento y exportación del detector (Ultralytics YOLO).
web/         Demo para el móvil: mismo pipeline en el navegador (foto, vídeo o cámara en vivo).
docs/        SAFETY.md (reglas) y SETUP_UNITY.md (montaje paso a paso).
```

## Probar en el móvil (sin visor)

`web/` es una versión web que usa el mismo tracker y las mismas reglas de seguridad (portadas a
JavaScript y testeadas contra los valores de C#). Haces una foto o grabas un vídeo con el móvil y la
publicidad desaparece. Usa un detector sin entrenar (YOLO-World guiado por texto: "billboard",
"advertisement poster"...) que corre entero en el navegador con onnxruntime-web.

```bash
python ml/scripts/export_world.py --out web/model   # genera el detector (fp16, en 2 trozos de 12 MB)
python web/build.py                                  # empaqueta web/dist
node --test web/test/core.test.js                    # tests del port a JS
```

La regla de distancia (2 m) no existe en el móvil: necesita el sensor de profundidad del Quest.
La cámara en vivo solo funciona si `web/dist` se sirve por HTTPS desde tu propio dominio
(GitHub Pages, por ejemplo); dentro de una página de Claude usa «Foto» o «Vídeo».

## Puesta en marcha (Quest)

1. **Prueba rápida sin entrenar** (las TV y los monitores hacen de "anuncio"):
   ```bash
   pip install -r ml/requirements.txt
   python ml/scripts/export.py --weights yolo11n.pt --coco-smoke-test
   ```
2. Monta la escena siguiendo [docs/SETUP_UNITY.md](docs/SETUP_UNITY.md) y haz *Build And Run* en el Quest.
3. Pulsa **A/X** para activarlo y mira una TV a más de 2 m.

## Entrenar el detector

La forma más sencilla es el cuaderno de Google Colab, que usa una GPU gratuita y hace todo el proceso:

**[Abrir en Colab](https://colab.research.google.com/github/jmendozapuche/clearview/blob/ccr-3d807d19-g1sbwv/ml/colab/ClearView_entrenar.ipynb)**
(`ml/colab/ClearView_entrenar.ipynb`)

| Datos | De dónde | Cuenta |
|-------|----------|--------|
| Vallas, carteles, personas, vehículos, sillas de ruedas, semáforos, señales | Open Images V7 (`fetch_openimages.py`) | No |
| Logos | LogoDet-3K en Kaggle, un dataset de Roboflow o un zip propio (`voc_to_yolo.py`, `normalize_yolo.py`) | Kaggle o Roboflow |
| Personas y vehículos que faltan en esas fotos | Autoetiquetado con YOLO11x (`pseudo_label_safety.py`) | No |

Los mismos pasos a mano, en un equipo con GPU:

```bash
python ml/scripts/fetch_openimages.py --split train --out raw/openimages
python ml/scripts/voc_to_yolo.py --src raw/LogoDet-3K --out raw/logos --single-class logo --max-images 8000
python ml/scripts/pseudo_label_safety.py --source raw/logos
python ml/scripts/pseudo_label_safety.py --source raw/openimages
python ml/scripts/prepare_dataset.py --config ml/configs/colab.yaml
python ml/scripts/train.py --data datasets/clearview/data.yaml --batch -1 --device 0 --resume
python ml/scripts/eval_safety.py --weights runs/clearview/train/weights/best.pt \
    --data datasets/clearview/data.yaml --config ml/configs/colab.yaml
python ml/scripts/export.py --weights runs/clearview/train/weights/best.pt --config ml/configs/colab.yaml \
    --web web/model-trained                       # y además el .onnx para Unity
python web/build.py --model web/model-trained     # web con el modelo entrenado
```

**Por qué el autoetiquetado:** los datasets de logos no marcan a las personas de sus fotos, y Open Images
tampoco las marca todas. Sin completarlas, el modelo aprendería que una persona es "fondo" y detectaría peor
justo lo que nunca hay que tapar.

**Qué métrica mirar:** `eval_safety.py` destaca el **recall de las clases de seguridad**. Un anuncio que no
se detecta es una molestia; una persona que no se detecta puede acabar tapada.

**Personas impresas en anuncios:** en Open Images, el 51 % de las personas en fotos de vallas y carteles
son fotos o dibujos dentro del anuncio. Por defecto cuentan como personas, así que ClearView deja esos
anuncios visibles. `--drop-safety-depictions` enseña al modelo a distinguirlas; úsalo solo si después
`eval_safety.py` confirma que la detección de personas reales sigue alta.

## Tests

```bash
dotnet test tests/ClearView.Core.Tests   # lógica de seguridad (C#)
python -m pytest ml/tests                # pipeline de datos y exportación (Python)
node --test web/test/core.test.js        # port a JS de la lógica de seguridad
```

## Estado y siguientes pasos

- [x] Pipeline completo: cámara → detección → seguridad → parches anclados al mundo
- [x] Reglas de seguridad con tests
- [x] Pipeline de datos y entrenamiento, con modo de prueba COCO
- [x] Demo web para el móvil con detector sin entrenar (YOLO-World)
- [ ] Compilar y probar en un Quest 3 real (las APIs de Meta e Inference Engine no se han compilado contra los paquetes reales)
- [ ] Dataset propio de publicidad urbana capturado con el Quest
- [ ] Evaluación offline de seguridad: % de personas o señales que acabarían tapadas, con vídeos grabados
- [ ] Relleno más realista (inpainting ligero) en lugar de la mezcla de 4 colores
