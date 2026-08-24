using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Components.Pages.Generate.ByImage
{
    /// <summary>
    /// Servicio para manejar estado del componente
    /// El servicio almacena el flujo del formulario
    /// </summary>
    public class GBI_StateMachine
    {

        /// <summary>
        /// Imagen a usar para generar la paleta de colores
        /// </summary>
        public byte[]? Image { get; private set; }

        public void SetImage(byte[] image)
        {
            Image = image;
            NotifyStateChanged();
        }

        /// <summary>
        /// Imagen Base 64 para mostrar una previsualización
        /// </summary>
        public string? ImagePreview { get; private set; }

        public void SetImagePreview(string image)
        {
            ImagePreview = image;
            NotifyStateChanged();
        }

        /// <summary>
        /// Colores seleccionados para 
        /// </summary>
        public List<string> Colors { get; private set; } = [];

        public void SetColors(List<string> colors)
        {
            Colors = colors;
            NotifyStateChanged();
        }

        public bool Loading { get;  private set; } = false;

        public void SetLoading(bool loading)
        {
            Loading = loading;
            NotifyStateChanged();
        }

        public bool Loaded { get; private set; }

        public void SetLoaded(bool loaded)
        {
            Loaded = loaded;
            NotifyStateChanged();
        }

        public async Task Reset()
        {
            this.Image = null;
            this.Colors = [];
            this.Loading = false;
            this.Loaded = false;
            NotifyStateChanged();
        }

        public event Action? OnChange;
        private void NotifyStateChanged() => OnChange?.Invoke();
    }
}
