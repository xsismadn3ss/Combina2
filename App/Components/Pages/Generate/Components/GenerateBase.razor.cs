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
    /// <summary>
    /// Estado de la máquina de estados para el
    /// componente base GenerateBase
    /// </summary>
    [Inject] protected GStateMachine State { get; set; } = default!;

    protected override void OnInitialized()
    {
        State.OnChange += HandleStateChanged;
    }

    private async void HandleStateChanged()
    {
        await OnStateChangedAsync();
        await InvokeAsync(StateHasChanged);
    }

    /// <summary>
    /// Hook virtual para que los hijos ejecuten lógica al cambiar el estado
    /// sin tener que re-suscribirse al evento. El base ya hace StateHasChanged.
    /// </summary>
    protected virtual Task OnStateChangedAsync() => Task.CompletedTask;

    public virtual void Dispose()
    {
        State.OnChange -= HandleStateChanged;
        GC.SuppressFinalize(this);
    }
}