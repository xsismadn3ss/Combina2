using System;
using System.Collections.Generic;
using System.Text;

namespace Combina2.Components.Pages.ByImage
{
    internal static class InjectGenerate
    {
        /// <summary>
        /// Inyectar servicios para la vista Generate By Image
        /// </summary>
        /// <param name="services"></param>
        /// <returns></returns>
        public static IServiceCollection AddGBI_Services(this IServiceCollection services)
        {
            // Inyectar servicio state machine
            services.AddSingleton<GStateMachine>();

            return services;
        }
    }
}
