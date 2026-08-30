namespace Combina2.Services;

public interface IHarmonyRepository
{
    Task<List<string>> GetHarmonyOptions();
}