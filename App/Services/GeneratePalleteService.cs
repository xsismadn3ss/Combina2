using Combina2.Models;

namespace Combina2.Services;

public class GeneratePalleteDummyService : IGeneratePalleteService
{
    public async Task<PalleteResponse> GetPalleteAsync(CreatePalleteRequest dto)
    {
        // Esperar 1.5 segundos
        await Task.Delay(1500);

        // Generar respuesta
        var response = new PalleteResponse
        {
            Harmony = "Dummy Harmony",
            Colors =
            [
                new ColorModel() { Value = "#A0A0", Role = "Primary" },
                new ColorModel() { Value = "#5F5F", Role = "Secondary" },
                new ColorModel() { Value = "#5E219F", Role = "Accent" },
            ],
        };
        return response;
    }
}
