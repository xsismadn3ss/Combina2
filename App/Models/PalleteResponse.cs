using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Models
{
    public class PalleteResponse
    {
        public string Harmony { get; set; } = string.Empty;
        public List<ColorModel> Colors { get; set; } = [];
    }
}
