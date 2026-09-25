using System;
using System.Collections.Generic;
using System.Text;
using Web.Services;

namespace Web.Components.Pages.Generate.Services
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
            services.AddScoped<GStateMachine>();
            services.AddScoped<IColorPonderanceService, ColorPonderanceService>();
            services.AddScoped<IHarmonyRepository, HarmonyDummyRepo>();
            services.AddScoped<IGeneratePalleteService, GeneratePalleteDummyService>();

            return services;
        }
    }
}
