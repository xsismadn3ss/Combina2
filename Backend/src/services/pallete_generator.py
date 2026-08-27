from src.schemas.palette import Color, ColorRole, HarmonyType, PaletteResponse
from src.utils.color_converter import ColorConverter

#SISTEMA EXPERTO 1: Generador de Paletas Armónicas (Motor de Reglas HSL)
#
class ExpertSystemPaletteGenerator:
    """
    Sistema Experto basado en Reglas Lógicas Geométricas sobre el circulo cromático HSL.
    Aplica trasformaciones fijas de matiz (Hue):
    Complementario: +180°
    Triádico: +120°, +240°
    Análogo: ±30°
    Monocromatico: variaciones de L (Lightness)
    """
    def generate_palette(self, base_hex_colors: list[str], harmony: HarmonyType) -> PaletteResponse:
        primary_hex = base_hex_colors[0]
        h, s, l = ColorConverter.hex_to_hsl(primary_hex)

        result_colors: list[Color] = [
            Color(value=primary_hex.upper(), role=ColorRole.PRIMARY.value)
        ]

        if harmony == HarmonyType.COMPLEMENTARY:
            comp_h = (h + 180.0) % 360.0
            comp_hex = ColorConverter.hsl_to_hex(comp_h, s, l)
            result_colors.append(Color(value=comp_hex.upper(), role=ColorRole.COMPLEMENTARY.value))

        elif harmony == HarmonyType.TRIADIC:
            triad1_h = (h + 120.0) % 360.0
            triad2_h = (h + 240.0) % 360.0
            result_colors.append(Color(value=ColorConverter.hsl_to_hex(triad1_h, s, l).upper(),
                                       role=ColorRole.TRIADIC_1.value))
            result_colors.append(Color(value=ColorConverter.hsl_to_hex(triad2_h, s, l).upper(),
                                       role=ColorRole.TRIADIC_2.value))

        elif harmony == HarmonyType.ANALOGOUS:
            ana1_h = (h - 30.0) % 360.0
            ana2_h = (h + 30.0) % 360.0
            result_colors.append(Color(value=ColorConverter.hsl_to_hex(ana1_h, s, l).upper(),
                                       role=ColorRole.ANALOGOUS_1.value))
            result_colors.append(Color(value=ColorConverter.hsl_to_hex(ana2_h, s, l).upper(),
                                       role=ColorRole.ANALOGOUS_2.value))

        elif harmony == HarmonyType.MONOCHROMATIC:
            #Variación de Luminosidad
            l_light = min(100.0, l + 25.0)
            l_dark = max(0.0, l - 25.0)
            result_colors.append(Color(value=ColorConverter.hsl_to_hex(h, s, l_light).upper(), role="light_tone"))
            result_colors.append(Color(value=ColorConverter.hsl_to_hex(h, s, l_dark).upper(), role="dark_tone"))

        return PaletteResponse(type=harmony.value, colors=result_colors)
