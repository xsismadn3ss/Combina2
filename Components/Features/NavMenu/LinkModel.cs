using System.ComponentModel.DataAnnotations;

namespace Combina2.Components.Features.NavMenu
{
    public sealed class LinkModel
    {
        [Required]
        public string Name { get; set; } = String.Empty;
        public string IconName { get; set; } = String.Empty;
        public string Href { get; set; } = "/";
    }
}
