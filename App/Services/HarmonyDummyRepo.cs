namespace Combina2.Services;

/// <summary>
/// Servicio Dummy para obtener opciones de harmonía de colores
/// </summary>
public class HarmonyDummyRepo : IHarmonyRepository
{
    public Task<List<string>> GetHarmonyOptions()
    {
        var options = new List<string>
        {
            "monocromatica",
            "analogico",
            "complementario"
        };
        return Task.FromResult(options);

    }
}