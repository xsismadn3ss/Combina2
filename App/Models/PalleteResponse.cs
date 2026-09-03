using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json.Serialization;

namespace Combina2.Models
{
    public class PalleteResponse
    {
        [JsonPropertyName("harmony")]
        public string Harmony { get; set; } = string.Empty;
        [JsonPropertyName("colors")]
        public List<ColorModel> Colors { get; set; } = [];
    }
}
