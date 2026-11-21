# Seguridad en ClearView

ClearView modifica lo que ves del mundo real, así que el criterio de diseño es uno solo:

> **Fail-open:** ante cualquier duda, error o situación anómala, la publicidad **se deja visible**.
> Ver un anuncio nunca es peligroso; no ver un obstáculo, sí.

El parche que tapa un anuncio no es negro ni opaco. Se pinta con los colores que rodean el anuncio
(borde izquierdo, derecho, superior e inferior), con bordes difuminados, para que parezca
"pared vacía" en lugar de un bloque que tape la vista.

## Reglas

Todas están en código puro y testeado (`unity/Assets/ClearView/Core`, tests en `tests/`).
Los valores por defecto son conservadores y se pueden ajustar en el Inspector de `ClearViewController`.

| # | Regla | Por defecto | Dónde |
|---|-------|-------------|-------|
| R1 | Si no llegan detecciones nuevas durante 1 s, se retiran todos los parches | 1.0 s | `FailSafe` |
| R2 | **Nunca se tapa un objeto de seguridad** (personas, bicis, coches, motos, buses, camiones, semáforos, señales). Si el parche toca la zona de uno, ampliada con un margen, se cancela entero | margen 10 % del objeto, mínimo 2 % de la imagen | `SafetyPolicy` |
| R2b | Los objetos de seguridad se aceptan con menos confianza que los anuncios: preferimos falsos positivos de "persona" | 0.20 frente a 0.45 | `DecoderConfig` |
| R3 | El parche cubre solo el interior del anuncio: el contorno físico (marco, poste, borde del cartel) sigue visible | 6 % por lado | `SafetyPolicy` |
| R4 | Un parche no puede ocupar más del 12 % de la imagen, y todos juntos no más del 25 %. Un anuncio enorme suele significar que está muy cerca | 12 % / 25 % | `SafetyPolicy` |
| R5 | No se tapa nada cuyo borde inferior esté en el 20 % inferior de la imagen (carteles de acera y otros obstáculos a ras de suelo) | y > 0.80 | `SafetyPolicy` |
| R6 | No se tapa nada a menos de 2 m. Si no se puede medir la distancia a la superficie, tampoco se tapa | 2.0 m | `SafetyPolicy.AllowsDistance` + `MaskProjector` |
| R7 | Un anuncio tiene que detectarse en 3 ciclos antes de taparse. Los parches aparecen en 0.3 s y desaparecen en 0.1 s | 3 hits | `AdTracker`, `MaskRenderer` |
| R8 | Si giras la cabeza rápido (normalmente para mirar el tráfico) o vas más rápido que andando, los parches desaparecen al instante y vuelven poco a poco | 60–120 °/s, 1.6–2.3 m/s | `MotionGuard` |
| R9 | Opacidad máxima del 90 %: si algo se clasifica mal, se sigue intuyendo detrás del parche | 0.90 | `SafetyConfig.MaxOpacity` |
| R10 | Si los FPS bajan de 60 durante más de 1 s (incomodidad, mareo), se apaga | 60 fps | `FailSafe` |
| R11 | Interruptor: botón A/X o menú (gesto de menú de la mano izquierda) apaga y enciende al instante. **Arranca apagado**: lo enciendes tú a conciencia | apagado | `ClearViewController` |
| R12 | Indicador de estado siempre visible: verde = tapando anuncios, ámbar = en pausa, nada = apagado. Siempre sabes si estás viendo la realidad alterada | — | `statusIndicator` |
| R13 | Cualquier excepción desactiva ClearView y deja el passthrough limpio | — | `ClearViewController.Fail` |
| R14 | En el visor no se permite usar la profundidad "falsa" de distancia fija, porque la regla R6 sería ficticia | — | `ClearViewController.Start` |

## Lo que ClearView NO garantiza

- **El detector se equivoca.** Puede confundir una señal de tráfico con un póster. R2b, R3, R9 y el
  entrenamiento con clases de seguridad reducen el riesgo, pero no lo eliminan.
- **Hay latencia.** La detección va a pocos Hz. Por eso los parches se anclan al mundo (no a la
  cabeza) y solo se tapan cosas estáticas: lo que se mueve (anuncios en autobuses) toca un vehículo
  y R2 lo descarta.
- **Quest 3 no es un dispositivo para andar por la calle.** Meta desaconseja usarlo en exteriores
  con tráfico, y su campo de visión y su passthrough son inferiores a la vista natural. Este prototipo
  es para **investigación en entornos controlados**: interiores, de pie o sentado, mirando carteles
  o pantallas. No lo uses cruzando calles, conduciendo ni en bici.
- **Privacidad.** El acceso a la cámara procesa todo en el dispositivo y no guarda ni envía imágenes.
  Mantenlo así.

## Cómo validar antes de cada salida de prueba

1. Activa `debugOutline` en `MaskRenderer`: los parches se dibujan con borde magenta para ver si
   están bien alineados.
2. Mira el log (`adb logcat -s Unity`): `skip.SafetyOverlap`, `tooClose`, `noSurface`... indican qué
   regla está actuando.
3. Coloca una persona delante de un cartel: el parche **tiene** que desaparecer.
4. Gira la cabeza rápido: los parches **tienen** que desaparecer al instante.
5. Pulsa A/X: todo **tiene** que desaparecer en el mismo frame.
