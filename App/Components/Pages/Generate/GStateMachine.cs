using Combina2.Components.Pages.Generate.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Components.Pages.Generate;


/// <summary>
/// Servicio para manejar estado del componente
/// El servicio almacena el flujo del formulario
/// </summary>
public class GStateMachine
{
    // Estado de la imagen cargada
    public FormState FormState { get; set; } = new FormState();
    public ResponseState ResponseState { get; set; } = new ResponseState();
    public async Task Reset()
    {
        FormState.Reset();
        ResponseState.Reset();
        NotifyStateChanged();
    }

    public event Action? OnChange;
    public void NotifyStateChanged() => OnChange?.Invoke();

    public void OnAfterRender()
    {
        NotifyStateChanged();
    }
}
