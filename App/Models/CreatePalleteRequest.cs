using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text;
using System.Text.Json.Serialization;

namespace Combina2.Models
{
    public class CreatePalleteRequest
    {
        [MinLength(1)]
        [JsonPropertyName("colors")]
        public List<string> Colors { get; set; } = [];
        [Required]
        [JsonPropertyName("harmony")]
        public string Harmony { get; set; } = string.Empty;
    }
}
