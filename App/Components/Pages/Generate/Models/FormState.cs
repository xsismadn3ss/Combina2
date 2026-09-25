using Combina2.Models;

namespace Combina2.Components.Pages.Generate.Models;

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

    /// <summary>
    /// Propiedad para almacenar el paso en el que esta el formulario
    /// <br/>
    /// 0 = Subir imagen 
    /// <br/>
    /// 1 = Seleccionar colores
    /// <br/>
    /// 2 = Seleccionar harmonia
    /// <br/>
    /// 3 = Enviar formulario y obtener respuesta
    /// </summary>
    public int Step { get; set; } = 0;

    public bool IsHarmonySelected =>
        !string.IsNullOrEmpty(Harmony);

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
        IsValidImage &&
        SelectedColors.Count > 1 &&
        IsHarmonySelected;

    public void NextStep()
    {
        Step += 1;
    }

    public void PreviousStep()
    {
        Step -= 1;
    }


    public void Reset()
    {
        ImageLoading = false;
        ImageLoaded = false;
        Image = null;
        ImagePreview = string.Empty;
        Colors = [];
        SelectedColors = [];
        Harmony = string.Empty;
        Step = 0;
    }
}