using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Components.Features.NavMenu
{
    internal class NavMenuState
    {
        public bool MenuVisible { get; private set; }

        public event Action? OnChange;

        public void ToggleMenu()
        {
            MenuVisible = !MenuVisible;
            OnChange?.Invoke();
        }

    }
}
