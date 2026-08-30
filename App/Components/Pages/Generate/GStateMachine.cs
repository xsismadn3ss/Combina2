using Combina2.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Components.Pages
{
    public sealed class FormState
    {
        public bool Loading { get; set; } = false;
        public bool Loaded { get; set; } = false;
        public byte[]? Image { get; set; }
        public string ImagePreview { get; set; } = string.Empty;

        public List<ColorPonderance> Colors { get; set; } = [];
        public List<ColorPonderance> SelectedColors { get; set; } = [];

        public void Reset()
        {
            Loading = false;
            Loaded = false;
            Image = null;
            ImagePreview = string.Empty;
            Colors = [];
            SelectedColors = [];
        }

        /// <summary>
        /// Devuelve true cuando la imagen no es luna y si tiene bytes
        /// </summary>
        public bool IsValidImage => Image != null && Image.Length > 0;

        /// <summary>
        /// Devuelve true cuando hay una imagen valida y hay más de dos colores seleccionados
        /// </summary>
        public bool IsValidForm =>
            IsValidImage && SelectedColors.Count > 1;
    }

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
}
