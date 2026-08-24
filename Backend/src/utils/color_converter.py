import colorsys

# Utilidades para Espacios de Color (Hex <-> RGB <-> HSL)
class ColorConverter:
    @staticmethod
    def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
        hex_str = hex_str.lstrip('#')
        if len(hex_str) != 6:
            raise ValueError(f"Código HEX inválido: {hex_str}")
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, rgb[0])),
            max(0, min(255, rgb[1])),
            max(0, min(255, rgb[2]))
        )

    @staticmethod
    def hex_to_hsl(hex_str: str) -> tuple[float, float, float]:
        r, g, b = ColorConverter.hex_to_rgb(hex_str)
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        return (h * 360.0, s * 100.0, l * 100.0)

    @staticmethod
    def hsl_to_hex(h: float, s: float, l: float) -> str:
        h = (h % 360.0) / 360.0
        s = max(0.0, min(100.0, s)) / 100.0
        l = max(0.0, min(100.0, l)) / 100.0
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return ColorConverter.rgb_to_hex((round(r * 255), round(g * 255), round(b * 255)))