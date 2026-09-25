using Web.Models;

namespace Web.Services;

public interface IGeneratePalleteService
{
    Task<PalleteResponse> GetPalleteAsync(CreatePalleteRequest dto);
}