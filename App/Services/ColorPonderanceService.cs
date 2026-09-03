using Combina2.Models;
using SkiaSharp;

namespace Combina2.Services;

public class ColorPonderanceService : IColorPonderanceService
{
    const int MaxDimension = 128;
    const byte Mask = 0xF0;

    public IEnumerable<ColorPonderance> GetPonderances(byte[] imageBytes)
    {
        using var image = SKImage.FromEncodedData(imageBytes);
        if (image is null)
            return [];

        int origW = image.Width;
        int origH = image.Height;
        float scale = MathF.Sqrt((float)(MaxDimension * MaxDimension) / (origW * origH));
        int targetW = Math.Max(1, (int)(origW * scale));
        int targetH = Math.Max(1, (int)(origH * scale));

        var info = new SKImageInfo(targetW, targetH, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var bitmap = new SKBitmap(info);
        using var canvas = new SKCanvas(bitmap!);
        var sampling = new SKSamplingOptions(SKFilterMode.Linear);
        canvas.DrawImage(image, new SKRect(0, 0, targetW, targetH), sampling);
        canvas.Flush();

        int totalPixels = targetW * targetH;
        var colorCounts = new Dictionary<int, int>();
        var span = bitmap.GetPixelSpan();
        int rowBytes = bitmap.RowBytes;

        for (int y = 0; y < targetH; y++)
        {
            int rowOffset = y * rowBytes;
            for (int x = 0; x < targetW; x++)
            {
                int i = rowOffset + x * 4;
                int key = ((span[i] & Mask) << 16) | ((span[i + 1] & Mask) << 8) | (span[i + 2] & Mask);
                colorCounts.TryGetValue(key, out int count);
                colorCounts[key] = count + 1;
            }
        }

        return colorCounts
            .Select(kvp => new ColorPonderance
            {
                Color = kvp.Key.ToString("X6"),
                Percentage = (double)kvp.Value / totalPixels * 100
            })
            .OrderByDescending(c => c.Percentage)
            .Take(10);
    }
}
