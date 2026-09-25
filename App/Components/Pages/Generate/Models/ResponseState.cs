using Combina2.Models;

namespace Combina2.Components.Pages.Generate.Models;

public sealed class ResponseState
{
    public bool Loading { get; set; } = false;
    public bool Loaded { get; set; } = false;

    public PalleteResponse? Response { get; set; } = null;

    public void Reset()
    {
        Loading = false;
        Loaded = false;
        Response = null;
    }
}