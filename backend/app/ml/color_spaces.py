"""Conversiones de espacios de color: Hex <-> RGB <-> HSL <-> CIELAB.

Las redes (fases 3-4) entrenan e infieren en CIELAB normalizado, espacio
perceptualmente uniforme apto para loss/distancias DeltaE/CIEDE2000.
"""

import colorsys

import numpy as np

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_D65 = np.array([95.047, 100.0, 108.883])


class ColorConverter:
    @staticmethod
    def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
        hex_str = hex_str.lstrip("#")
        if len(hex_str) != 6:
            raise ValueError(f"Código HEX inválido: {hex_str}")
        try:
            return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        except ValueError:
            raise ValueError(f"Código HEX inválido: {hex_str}") from None

    @staticmethod
    def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return f"#{max(0, min(255, rgb[0])):02x}{max(0, min(255, rgb[1])):02x}{max(0, min(255, rgb[2])):02x}"

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
        return ColorConverter.rgb_to_hex(
            (round(r * 255), round(g * 255), round(b * 255))
        )

    # --- CIELAB (D65) ---

    @staticmethod
    def rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
        arr = np.asarray(rgb, dtype=np.float64) / 255.0
        arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
        xyz = _RGB_TO_XYZ @ arr * 100.0
        return ColorConverter.xyz_to_lab(tuple(xyz))

    @staticmethod
    def xyz_to_lab(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        v = np.asarray(xyz, dtype=np.float64) / _D65

        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

        fx, fy, fz = f(v)
        return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))

    @staticmethod
    def lab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
        l, a, b = lab
        fy = (l + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0

        def finv(t: float) -> float:
            t3 = t**3
            return t3 if t3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

        xyz = np.array([finv(fx), finv(fy), finv(fz)]) * _D65 / 100.0
        arr = _XYZ_TO_RGB @ xyz
        arr = np.where(arr <= 0.0031308, 12.92 * arr, 1.055 * arr ** (1.0 / 2.4) - 0.055)
        return tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in arr)  # type: ignore[return-value]

    @staticmethod
    def hex_to_lab(hex_str: str) -> tuple[float, float, float]:
        return ColorConverter.rgb_to_lab(ColorConverter.hex_to_rgb(hex_str))

    @staticmethod
    def lab_to_hex(lab: tuple[float, float, float]) -> str:
        return ColorConverter.rgb_to_hex(ColorConverter.lab_to_rgb(lab))

    # --- Normalización para redes: L∈[0,1], a/b∈[-1,1] aprox. ---

    @staticmethod
    def lab_normalize(lab: tuple[float, float, float]) -> tuple[float, float, float]:
        l, a, b = lab
        return (l / 100.0, max(-1.0, min(1.0, a / 128.0)), max(-1.0, min(1.0, b / 128.0)))

    @staticmethod
    def lab_denormalize(nlab: tuple[float, float, float]) -> tuple[float, float, float]:
        l, a, b = nlab
        return (l * 100.0, a * 128.0, b * 128.0)
