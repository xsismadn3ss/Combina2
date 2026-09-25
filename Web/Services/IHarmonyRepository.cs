namespace Web.Services;

public interface IHarmonyRepository
{
    Task<List<string>> GetHarmonyOptions();
}