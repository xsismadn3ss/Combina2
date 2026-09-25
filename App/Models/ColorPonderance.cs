namespace Combina2.Models;

/// <summary>
/// Modelo de catos para representar un color y el porcentaje
/// que ocupa en una imagen
/// </summary>
public class ColorPonderance
{
    /// <summary>
    /// Codigo hexadecimal que representa al color
    /// </summary>
    public string Color = string.Empty;

    /// <summary>
    /// Porcentaje de color que representa este objeto en la imagen
    /// </summary>
    public double Percentage = 0;
}
