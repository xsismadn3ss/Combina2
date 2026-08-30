using Combina2.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Components.Pages.Generate
{
    /// <summary>
    /// Estado del formulario para generar una paleta de colores
    /// </summary>
    public sealed class FormState
    {
        public bool ImageLoading { get; set; } = false;
        public bool ImageLoaded { get; set; } = false;
        /// <summary>
        /// Arreglo de bytes de la imagen cargada
        /// </summary>
        public byte[]? Image { get; set; }
        /// <summary>
        /// String Base64 de la imagen para mostrar como preview
        /// </summary>
        public string ImagePreview { get; set; } = string.Empty;

        /// <summary>
        /// Colores frecuentes en la imagen
        /// </summary>
        public List<ColorPonderance> Colors { get; set; } = [];
        /// <summary>
        /// Colores seleccionados
        /// </summary>
        public List<string> SelectedColors { get; set; } = [];

        /// <summary>
        /// Harmonia seleccionada para generar una paleta de colores
        /// </summary>
        public string Harmony { get; set; } = string.Empty;

        public void Reset()
        {
            ImageLoading = false;
            ImageLoaded = false;
            Image = null;
            ImagePreview = string.Empty;
            Colors = [];
            SelectedColors = [];
        }

        /// <summary>
        /// Devuelve true cuando la imagen no es luna y si tiene bytes
        /// </summary>
        public bool IsValidImage => Image != null && Image.Length > 0;

        public bool CanSelectHarmony =>
            IsValidImage && SelectedColors.Count > 1;

        /// <summary>
        /// Devuelve true cuando hay una imagen valida y hay más de dos colores seleccionados
        /// </summary>
        public bool IsValidForm =>
            IsValidImage && SelectedColors.Count > 1 && !string.IsNullOrEmpty(Harmony);
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
