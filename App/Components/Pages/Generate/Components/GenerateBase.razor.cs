using Combina2.Components.Pages.Generate.Services;
using Microsoft.AspNetCore.Components;

namespace Combina2.Components.Pages.Generate.Components;

/// <summary>
/// Componente base que inyecta el servicio GStateMachine
/// <br/>
/// Este componente permite que los componente que hereden
/// esta clase tengan disponible el servicio para usarlo
/// </summary>
public partial class GenerateBase : ComponentBase, IDisposable
{
    [Inject] protected GStateMachine State { get; set; } = default!;

    protected override void OnInitialized()
    {
        State.OnChange += StateHasChanged;
    }

    public void Dispose()
    {
        State.OnChange -= StateHasChanged;
    }
}