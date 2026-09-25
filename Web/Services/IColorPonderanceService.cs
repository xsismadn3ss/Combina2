using Web.Models;

namespace Web.Services;

/// <summary>
/// Servicio para obtener porcentaje de ocurrencias de colores
/// dentro de una imagen
/// </summary>
public interface IColorPonderanceService
{
    IEnumerable<ColorPonderance> GetPonderances(byte[] Image);
}
