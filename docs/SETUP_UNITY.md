# Montar el prototipo en Unity para Quest 3 / 3S

> El código de `unity/Assets/ClearView` se ha comprobado en compilación contra las librerías de
> referencia de Unity, pero **las APIs de Meta y de Inference Engine se escribieron según su
> documentación y no se han compilado contra los paquetes reales**. Si algún nombre ha cambiado en tu
> versión, los únicos archivos que tocan esas APIs son `QuestPassthroughFrameSource.cs`,
> `DepthProviders.cs` (MRUK), `AdDetector.cs` (Inference Engine) y el `HandleKillSwitch` de
> `ClearViewController.cs` (OVRInput).

## Requisitos

- Quest 3 o 3S con Horizon OS v74 o superior (necesario para la Passthrough Camera API).
- Unity 6 (6000.0 LTS o posterior) con *Android Build Support*.
- Modo desarrollador activado en el visor.

## 1. Proyecto y paquetes

1. Crea un proyecto **Universal 3D** (URP).
2. Package Manager → instala:
   - **Meta XR Core SDK** (`com.meta.xr.sdk.core`)
   - **Meta MR Utility Kit** (`com.meta.xr.mrutilitykit`), que aporta `EnvironmentRaycastManager` (Depth API)
   - **Inference Engine** (`com.unity.ai.inference`). Si usas Sentis 2.1, cambia
     `using Unity.InferenceEngine;` por `using Unity.Sentis;` en `AdDetector.cs` y `ClearViewController.cs`.
3. Clona `github.com/oculus-samples/Unity-PassthroughCameraApiSamples` y copia a tu proyecto la
   carpeta con `WebCamTextureManager`, `PassthroughCameraUtils` y los permisos
   (`Assets/PassthroughCameraApiSamples/PassthroughCamera`).
4. Copia `unity/Assets/ClearView` de este repo dentro de `Assets/`.
5. *Project Settings → Player → Android → Scripting Define Symbols*: añade
   `CLEARVIEW_META;CLEARVIEW_MRUK`.
6. *Meta → Tools → Project Setup Tool*: aplica todos los *Fix*. Comprueba que el manifiesto incluye
   `horizonos.permission.HEADSET_CAMERA` y `com.oculus.permission.USE_SCENE`.

## 2. Modelo

Prueba rápida sin entrenar (las **TV/monitores** hacen de "anuncio"):

```bash
pip install -r ml/requirements.txt
python ml/scripts/export.py --weights yolo11n.pt --coco-smoke-test
```

Esto deja `clearview.onnx` y `clearview_labels.txt` en `unity/Assets/ClearView/Model/`.
Cópialos a la misma ruta de tu proyecto. Unity importa el `.onnx` como `ModelAsset`.

Para el modelo real de publicidad, consulta el README (sección *Entrenar el detector*).

## 3. Escena

1. Borra la *Main Camera*. Añade **OVRCameraRig** y, en `OVRManager`, activa *Passthrough Support*,
   *Scene Support* y *Passthrough Camera Access*.
2. Añade un **OVRPassthroughLayer** (*Placement: Underlay*). En la cámara de `CenterEyeAnchor`:
   *Background Type = Solid Color* y color `(0,0,0,0)`.
3. Añade el prefab o componente **WebCamTextureManager** de los samples de Meta.
4. Añade **EnvironmentRaycastManager** (MRUK). Si tu versión lo pide, también `EnvironmentDepthManager`.
5. Crea un GameObject vacío **ClearView** en el origen, con:
   - `QuestPassthroughFrameSource`: arrastra el `WebCamTextureManager`.
   - `MrukDepthProvider`: arrastra el `EnvironmentRaycastManager`.
   - `MaskRenderer`: un material nuevo con el shader **ClearView/SoftFill**.
   - `ClearViewController`:
     - *Frame Source*, *Depth Provider* y *Mask Renderer*: los componentes anteriores
     - *Head*: `CenterEyeAnchor`
     - *Model*: `clearview.onnx`
     - *Labels*: `clearview_labels.txt`
     - *Input* 416 × 320
6. Indicador de estado: una esfera de escala `0.01`, hija de `CenterEyeAnchor` en `(-0.12, -0.10, 0.5)`,
   con material *Unlit*. Arrástrala a *Status Indicator*.

## 4. Primera prueba

1. *Build And Run* en el visor. Acepta el permiso de cámara (y de escena).
2. Pulsa **A** o **X** para activarlo. El indicador se pone verde.
3. Mira una TV o un monitor a más de 2 m: debería "desaparecer" con un parche del color de la pared.
4. Logs: `adb logcat -s Unity | grep ClearView`.

## Calibración (si los parches salen desplazados)

| Síntoma | Ajuste |
|---------|--------|
| Parche desplazado verticalmente en espejo (arriba ↔ abajo) | `Decoder → Flip Y` |
| Colores del parche de arriba y abajo invertidos | `Color Rows Bottom Up` |
| El parche es más oscuro o más claro que la pared | `Color Gain` |
| Parpadeo | Sube `Tracker → Min Hits To Confirm` o baja `Position Follow Rate` |
| Caída de FPS | Baja `Layers Per Frame` |

Usa `debugOutline` en `MaskRenderer` mientras calibras.

## Probar en el Editor sin visor

Sustituye `QuestPassthroughFrameSource` por `EditorWebcamFrameSource` (webcam del PC) y
`MrukDepthProvider` por `FixedDistanceDepthProvider`. Pulsa Espacio para activarlo.
