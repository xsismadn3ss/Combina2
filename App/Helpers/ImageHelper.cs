using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Helpers
{
    internal static class ImageHelper
    {
        public static byte[] CompressImage(byte[] originalImage, int maxSizeBytes = 150_000)
        {
            using var input = new SKMemoryStream(originalImage);
            using var bitmap = SKBitmap.Decode(input);
            using var image = SKImage.FromBitmap(bitmap);

            int quality = 85; // calidad inicial
            byte[] result;

            do
            {
                using var data = image.Encode(SKEncodedImageFormat.Jpeg, quality);
                result = data.ToArray();

                if (result.Length <= maxSizeBytes || quality <= 30)
                    break;

                quality -= 10; // bajar calidad en pasos de 10
            }
            while (true);

            return result;
        }
    }
}

