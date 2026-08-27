using Combina2.Models;

namespace Combina2.Services;

public interface IColorPonderanceService
{
    IEnumerable<ColorPonderance> GetPonderances(byte[] Image);
}
