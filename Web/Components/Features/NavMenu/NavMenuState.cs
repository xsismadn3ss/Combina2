using System;
using System.Collections.Generic;
using System.Text;

namespace Web.Components.Features.NavMenu
{
    public class NavMenuState
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
