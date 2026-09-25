using Combina2.Models;

namespace Combina2.Services;

public interface IGeneratePalleteService
{
    Task<PalleteResponse> GetPalleteAsync(CreatePalleteRequest dto);
}