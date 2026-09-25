using System;
using System.Collections.Generic;
using System.Text;
using Combina2.Services;

namespace Combina2.Components.Pages.Generate.Services
{
    internal static class InjectGenerate
    {
        /// <summary>
        /// Inyectar servicios para la vista Generate
        /// </summary>
        /// <param name="services">Colecccion de servicios</param>
        /// <returns></returns>
        public static IServiceCollection AddGBI_Services(this IServiceCollection services)
        {
            // Inyectar servicio state machine
            services.AddSingleton<GStateMachine>();
            services.AddSingleton<IColorPonderanceService, ColorPonderanceService>();
            services.AddSingleton<IHarmonyRepository, HarmonyDummyRepo>();
            services.AddSingleton<IGeneratePalleteService, GeneratePalleteDummyService>();

            return services;
        }
    }
}
