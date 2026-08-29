using Combina2.Models;
using SkiaSharp;

namespace Combina2.Services;

public class ColorPonderanceService : IColorPonderanceService
{
    public IEnumerable<ColorPonderance> GetPonderances(byte[] imageBytes)
    {
        using var bitmap = SKBitmap.Decode(imageBytes);

        var colorCounts = new Dictionary<string, int>();
        int totalPixels = bitmap.Width * bitmap.Height;

        // Recorrer pixeles para contarlos uno a uno
        foreach (var pixel in bitmap.Pixels)
        {
            string hex = $"{pixel.Red:X2}{pixel.Green:X2}{pixel.Blue:X2}";

            if (colorCounts.TryGetValue(hex, out int value))
            {
                // Aumentar una unidad cuando ya existe la clave en el diccionario
                colorCounts[hex] = ++value;
            }
            else
            {
                // Declarar la clave en el diccionario
                colorCounts[hex] = 1;
            }


        }
        // Calcular ponderancias
        var ponderances = colorCounts.Select(
            kvp => new ColorPonderance
            {
                Color = kvp.Key,
                Percentage = (double)kvp.Value / totalPixels * 10
            }).OrderByDescending(c => c.Percentage);

        // Obtener solo los primeros diez más frecuentes
        return ponderances.Take(10);
    }
}
